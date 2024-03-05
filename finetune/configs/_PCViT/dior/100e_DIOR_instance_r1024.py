'''
train 5862 | 32634
{'airplane': 1238, 'airport': 328, 'baseball_field': 1245, 'basketball_court': 561, 
 'bridge': 506, 'chimney': 342, 'dam': 252, 'expressway_service_area': 537, 'expressway_toll_station': 292, 
 'golf_field': 237, 'ground_track_field': 565, 'harbor': 1147, 'overpass': 577, 'ship': 13028, 'stadium': 293, 
 'storage_tank': 1337, 'tennis_court': 2233, 'train_station': 246, 'vehicle': 6709, 'wind_mill': 961}

val 5863 | 35439
{'airplane': 650, 'airport': 334, 'baseball_field': 1139, 'basketball_court': 516, 
 'bridge': 861, 'chimney': 307, 'dam': 260, 'expressway_service_area': 543, 'expressway_toll_station': 318, 
 'golf_field': 274, 'ground_track_field': 597, 'harbor': 1217, 'overpass': 753, 'ship': 14323, 'stadium': 302, 
 'storage_tank': 1705, 'tennis_court': 2665, 'train_station': 255, 'vehicle': 7016, 'wind_mill': 1404}
'''


_base_ = '../../_base_/default_runtime.py'
# dataset settings
dataset_type = 'DIORDataset'
data_root = 'data/DIOR-R/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)

file_client_args = dict(backend='disk')
# comment out the code below to use different file client
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))

# train_pipeline = [
#     dict(type='LoadImageFromFile', file_client_args=file_client_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='Resize',
#         img_scale=image_size,
#         ratio_range=(0.1, 2.0),
#         multiscale_mode='range',
#         keep_ratio=True),
#     dict(
#         type='RandomCrop',
#         crop_type='absolute_range',
#         crop_size=image_size,
#         recompute_bbox=True,
#         allow_negative_crop=True),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=image_size),  # padding to image_size leads 0.5+ mAP
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=image_size, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1024),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    # train=dict(
    #     type='RepeatDataset',
    #     times=4,  # simply change this from 2 to 16 for 50e - 400e training.
    #     dataset=dict(
    #         type=dataset_type,
    #         ann_file="/home/data/tianpenghao/DIOR-R/DIOR_train_coco.json",
    #         img_prefix="/home/data/tianpenghao/DIOR-R/JPEGImages-trainval/",
    #         pipeline=train_pipeline)),
    train=dict(
        type=dataset_type,
        ann_file="/home/data/jiyuqing/dataset/DIOR-R/DIOR_train.json",
        img_prefix="/home/data/jiyuqing/dataset/DIOR-R/JPEGImages-trainval/",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file="/home/data/jiyuqing/dataset/DIOR-R/DIOR_val.json",
        img_prefix="/home/data/jiyuqing/dataset/DIOR-R/JPEGImages-trainval/",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file="/home/data/jiyuqing/dataset/DIOR-R/dior_test_visual.json",
        # img_prefix="/home/data/jiyuqing/dataset/DIOR-R/dior_test_visual/",
        ann_file="/home/data/jiyuqing/dataset/DIOR-R/DIOR_val.json",
        img_prefix="/home/data/jiyuqing/dataset/DIOR-R/JPEGImages-trainval/",
        # ann_file="/home/data/jiyuqing/dataset/DIOR-R/DIOR_test.json",
        # img_prefix="/home/data/jiyuqing/dataset/DIOR-R/JPEGImages-test/",
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric=['bbox', 'segm'])
# evaluation = dict(interval=3, metric='bbox', save_best='auto')
evaluation = dict(start=3, interval=1, metric='bbox', save_best='auto')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)


# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)


# # optimizer
# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[48, 66])
# runner = dict(type='EpochBasedRunner', max_epochs=72)
