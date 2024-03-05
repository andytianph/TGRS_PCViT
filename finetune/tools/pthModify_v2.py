import os
import torch
import torchvision.models as models
import torch.nn as nn


from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '8' 

###   Padding zerp 2d   ###
# input = torch.randn(4, 5, 1, 1)
# print(input)

# pad_len1 = nn.ZeroPad2d(1)
# output = pad_len1(input)
# print(output)


# state_dict = torch.load('/home/data/jiyuqing/.tph/tph_vitdet/MAEPretrain_SceneClassification/output_dir/millionAID_224/1600_0.42_0.00015_0.05_512_second/checkpoint-720.pth')
state_dict = torch.load('/home/data/jiyuqing/.tph/tph_vitdet/work_dirs/pretrained/vitae-b-checkpoint-1599-transform-no-average.pth')
state_dict_vis = torch.load('/home/data/jiyuqing/.tph/tph_vitdet/MAEPretrain_SceneClassification/output_dir/millionAID_224/1600_0.42_0.00015_0.05_512_first/checkpoint-120.pth')
# print(state_dict)

pad_len1 = nn.ZeroPad2d(1)

# print(state_dict['model']['blocks.0.PCM.0.weight'])
state_dict['model']['blocks.0.PCM.0.weight'] = state_dict['model']['blocks.0.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
# print(state_dict['model']['blocks.0.PCM.0.weight'])
state_dict['model']['blocks.0.PCM.3.weight'] = state_dict['model']['blocks.0.PCM.3.weight'][:,:,1,1].view(768,16,1,1)


state_dict['model']['blocks.1.PCM.0.weight'] = state_dict['model']['blocks.1.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.1.PCM.3.weight'] = state_dict['model']['blocks.1.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.2.PCM.0.weight'] = state_dict['model']['blocks.2.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.2.PCM.3.weight'] = state_dict['model']['blocks.2.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.3.PCM.0.weight'] = state_dict['model']['blocks.3.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.3.PCM.3.weight'] = state_dict['model']['blocks.3.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.4.PCM.0.weight'] = state_dict['model']['blocks.4.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.4.PCM.3.weight'] = state_dict['model']['blocks.4.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.5.PCM.0.weight'] = state_dict['model']['blocks.5.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.5.PCM.3.weight'] = state_dict['model']['blocks.5.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.6.PCM.0.weight'] = state_dict['model']['blocks.6.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.6.PCM.3.weight'] = state_dict['model']['blocks.6.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.7.PCM.0.weight'] = state_dict['model']['blocks.7.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.7.PCM.3.weight'] = state_dict['model']['blocks.7.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.8.PCM.0.weight'] = state_dict['model']['blocks.8.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.8.PCM.3.weight'] = state_dict['model']['blocks.8.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.9.PCM.0.weight'] = state_dict['model']['blocks.9.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.9.PCM.3.weight'] = state_dict['model']['blocks.9.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.10.PCM.0.weight'] = state_dict['model']['blocks.10.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.10.PCM.3.weight'] = state_dict['model']['blocks.10.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['blocks.11.PCM.0.weight'] = state_dict['model']['blocks.11.PCM.0.weight'][:,:,1,1].view(3072,4,1,1)
state_dict['model']['blocks.11.PCM.3.weight'] = state_dict['model']['blocks.11.PCM.3.weight'][:,:,1,1].view(768,16,1,1)

state_dict['model']['proj_embed.0.weight'] = state_dict_vis['model']['proj_embed.0.weight']
state_dict['model']['proj_embed.0.bias'] = state_dict_vis['model']['proj_embed.0.bias']
state_dict['model']['proj_embed.1.weight'] = state_dict_vis['model']['proj_embed.1.weight']
state_dict['model']['proj_embed.1.bias'] = state_dict_vis['model']['proj_embed.1.bias']
state_dict['model']['proj_embed.3.weight'] = state_dict_vis['model']['proj_embed.3.weight']
state_dict['model']['proj_embed.3.bias'] = state_dict_vis['model']['proj_embed.3.bias']
state_dict['model']['proj_embed.4.weight'] = state_dict_vis['model']['proj_embed.4.weight']
state_dict['model']['proj_embed.4.bias'] = state_dict_vis['model']['proj_embed.4.bias']
state_dict['model']['proj_embed.6.weight'] = state_dict_vis['model']['proj_embed.6.weight']
state_dict['model']['proj_embed.6.bias'] = state_dict_vis['model']['proj_embed.6.bias']
state_dict['model']['predictor.0.weight'] = state_dict_vis['model']['predictor.0.weight']
state_dict['model']['predictor.0.bias'] = state_dict_vis['model']['predictor.0.bias']


torch.save(state_dict, '/home/data/jiyuqing/.tph/tph_vitdet/work_dirs/pretrained/vitae-b-checkpoint-1599_3072_4_1_1_JD.pth')

# new_dict = torch.load('/home/data/jiyuqing/.tph/tph_vitdet/work_dirs/pretrained/vitae-b-checkpoint-1599_3072_4_1_1.pth')
print("done")