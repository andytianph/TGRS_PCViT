import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from fairscale.nn.checkpoint import checkpoint_wrapper

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.layers import DropPath, Mlp, trunc_normal_

from .ViTAE.NormalCell_knn import NormalCell_knn

# from mmcv_custom import load_checkpoint
from mmdet.models.backbones.utils.checkpoint import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES

import warnings



class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            with warnings.catch_warnings(record=True):
                if x.numel() == 0 and self.training:
                    # https://github.com/pytorch/pytorch/issues/12013
                    assert not isinstance(
                        self.norm, torch.nn.SyncBatchNorm
                    ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": torch.nn.BatchNorm2d,
            # # Fixed in https://github.com/pytorch/pytorch/pull/36382
            # "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            # "GN": lambda channels: nn.GroupNorm(32, channels),
            # # for debugging:
            # "nnSyncBN": nn.SyncBatchNorm,
            # "naiveSyncBN": NaiveSyncBatchNorm,
            # # expose stats_mode N as an option to caller, required for zero-len inputs
            # "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(
                self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()))))
        else:
            x = x + self.drop_path(
                self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()))
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x), (Hp, Wp)


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1).contiguous()

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# class PCT_Block(nn.Module):
#     """Transformer blocks with support of window attention and residual propagation blocks"""
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         drop_path=0.0,
#         norm_layer=nn.LayerNorm,
#         act_layer=nn.GELU,
#         use_rel_pos=False,
#         rel_pos_zero_init=True,
#         window_size=0,
#         input_size=None,
#     ):
#         """
#         Args:
#             dim (int): Number of input channels.
#             num_heads (int): Number of attention heads in each ViT block.
#             mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#             qkv_bias (bool): If True, add a learnable bias to query, key, value.
#             drop_path (float): Stochastic depth rate.
#             norm_layer (nn.Module): Normalization layer.
#             act_layer (nn.Module): Activation layer.
#             use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
#             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
#             window_size (int): Window size for window attention blocks. If it equals 0, then not
#                 use window attention.
#             input_size (int or None): Input resolution for calculating the relative positional
#                 parameter size.
#         """
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size if window_size == 0 else (window_size, window_size),
#         )

#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

#         self.window_size = window_size

#         PCM_dim = int(dim * mlp_ratio)
#         self.PCM = nn.Sequential(
#                             nn.Conv2d(dim, PCM_dim, 3, 1, 3//2, 1, 1),
#                             nn.BatchNorm2d(PCM_dim),
#                             nn.SiLU(inplace=True),
#                             nn.Conv2d(PCM_dim, dim, 3, 1, 3//2, 1, 1),
#                             )

#     def forward(self, x):
#         shortcut = x

#         x_2d = x.permute(0, 3, 1, 2).contiguous()
#         convX = self.drop_path(self.PCM(x_2d).permute(0, 2, 3, 1).contiguous())
        
#         x = self.norm1(x)
#         # Window partition
#         if self.window_size > 0:
#             H, W = x.shape[1], x.shape[2]
#             x, pad_hw = window_partition(x, self.window_size)

#         x = self.attn(x)
#         # Reverse window partition
#         if self.window_size > 0:
#             x = window_unpartition(x, self.window_size, pad_hw, (H, W))

#         x = shortcut + self.drop_path(x)
#         x = x + convX
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


@BACKBONES.register_module()
class ConvViTAE_knn(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=(4, 2, 2),
        in_chans=3,
        embed_dim=(256, 384, 768),
        depth=(2, 2, 11),
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        # norm_layer=nn.LayerNorm,
        norm="LN",
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        pretrained=None
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed1 = PatchEmbed(
            patch_size=patch_size[0],
            in_chans=in_chans,
            embed_dim=embed_dim[0],
        )
        self.patch_embed2 = PatchEmbed(
            patch_size=patch_size[1],
            in_chans=embed_dim[0],
            embed_dim=embed_dim[1],
        )
        self.patch_embed3 = PatchEmbed(
            img_size = 1024 // 4 // 2,
            patch_size=patch_size[2],
            in_chans=embed_dim[1],
            embed_dim=embed_dim[2],
        )
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // 16) * (pretrain_img_size // 16)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim[-1]))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer)
            for i in range(depth[0])])

        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i + depth[0]],
                act_layer=act_layer,
                norm_layer=norm_layer)
            for i in range(depth[1])])

        self.blocks3 = nn.ModuleList()
        for i in range(depth[2]):
            block = NormalCell_knn(
                dim=embed_dim[2],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=dpr[depth[0] + depth[1] + i],
                norm_layer=norm_layer,
                class_token=False,
                group=embed_dim[2]//4,
                window_size=(14, 14) if ((i + 2) % 3 != 0) else (64, 64),
                window=((i + 2) % 3 != 0),
                )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks3.append(block)

        self._out_feature_channels = {out_feature: embed_dim[-1]}
        self._out_feature_strides = {out_feature: 16}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.pretrained = pretrained

        use_bias = norm == ""
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            get_norm(norm, 384),
            nn.GELU(),
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            Conv2d(192, 256, kernel_size=1, bias=use_bias, 
                    norm=get_norm(norm, 256),),
            Conv2d(256, 256, kernel_size=1, bias=use_bias, 
                    norm=get_norm(norm, 256),),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            Conv2d(384, 256, kernel_size=1, bias=use_bias, 
                    norm=get_norm(norm, 256),),
            Conv2d(256, 256, kernel_size=1, bias=use_bias, 
                    norm=get_norm(norm, 256),),
        )

        # self.fpn3 = nn.Identity()
        self.fpn3 = nn.Sequential(
            Conv2d(768, 256, kernel_size=1, bias=use_bias, 
                    norm=get_norm(norm, 256),),
            Conv2d(256, 256, kernel_size=1, bias=use_bias, 
                    norm=get_norm(norm, 256),),
        )

        # self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(768, 256, kernel_size=1, bias=use_bias, 
                    norm=get_norm(norm, 256),),
            Conv2d(256, 256, kernel_size=1, bias=use_bias, 
                    norm=get_norm(norm, 256),),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = pretrained or self.pretrained
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            print(f"load from {pretrained}")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        x, (Hp, Wp) = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)

        x, (Hp, Wp) = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        x, (Hp, Wp) = self.patch_embed3(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.patch_embed4(x)

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        b, h, w, c = x.shape
        x = x.view(b, h*w, c)
        for blk in self.blocks3:
            blk.H = Hp
            blk.W = Wp
            x = blk(x)
        x = x.view(b, h, w, c)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2).contiguous()}
        
        features = []
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            features.append(ops[i](outputs['last_feat']))

        return tuple(features)


# def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
#     """
#     Calculate lr decay rate for different ViT blocks.
#     Args:
#         name (string): parameter name.
#         lr_decay_rate (float): base lr decay rate.
#         num_layers (int): number of ViT blocks.

#     Returns:
#         lr decay rate for the given parameter.
#     """
#     layer_id = num_layers + 1
#     if name.startswith("backbone"):
#         if ".patch_embed" in name or ".blocks1." in name or ".blocks2." in name:
#             layer_id = 0
#         elif ".blocks3." in name and ".residual." not in name:
#             layer_id = int(name[name.find(".blocks3.") :].split(".")[2]) + 1

#     return lr_decay_rate ** (num_layers + 1 - layer_id)
