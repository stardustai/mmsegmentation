norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        with_cp=True),
    decode_head=[
        dict(
            num_classes=7,
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=720,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            dropout_ratio=-1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4,
                class_weight=[
                    2.93173, 0.658505, 1, 1, 6.421162, 2.968377, 0.47459
                ])),
        dict(
            num_classes=7,
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=dict(type='BN', requires_grad=True),
            dropout_ratio=-1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[
                    2.93173, 0.658505, 1, 1, 6.421162, 2.968377, 0.47459
                ]))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'ParkinglotDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[86.243, 84.9454, 81.4281],
    std=[43.0072, 42.2812, 42.9458],
    to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1800, 1800), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[86.243, 84.9454, 81.4281],
        std=[43.0072, 42.2812, 42.9458],
        to_rgb=True),
    dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, img_scale=(1800, 1800)),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[86.243, 84.9454, 81.4281],
                std=[43.0072, 42.2812, 42.9458],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='ParkinglotDataset',
        data_root='data/parkinglot/',
        img_dir='images',
        ann_dir='labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(1800, 1800), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[86.243, 84.9454, 81.4281],
                std=[43.0072, 42.2812, 42.9458],
                to_rgb=True),
            dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split='train.csv'),
    val=dict(
        type='ParkinglotDataset',
        data_root='data/parkinglot/',
        img_dir='images',
        ann_dir='labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', keep_ratio=True,
                        img_scale=(1800, 1800)),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[86.243, 84.9454, 81.4281],
                        std=[43.0072, 42.2812, 42.9458],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='val.csv'),
    test=dict(
        type='ParkinglotDataset',
        data_root='data/parkinglot/',
        img_dir='images',
        ann_dir='labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', keep_ratio=True,
                        img_scale=(1800, 1800)),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[86.243, 84.9454, 81.4281],
                        std=[43.0072, 42.2812, 42.9458],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='val.csv'),
    data_root='data/parkinglot/')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'
resume_from = 'checkpoints/Parkinglot_ocr_hr_norm_cw/latest.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='SGD',
    lr=0.0003,
    momentum=0.9,
    weight_decay=1e-05,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=2))))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.0)
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=640000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=5)
evaluation = dict(interval=10000, metric='mIoU')
work_dir = 'checkpoints/Parkinglot_ocr_hr_norm_cw/'
base = 'data/parkinglot/'
palette = dict(
    road=(0, 0, 0),
    curb=(0, 255, 255),
    obstacle=(0, 255, 0),
    chock=(255, 0, 0),
    parking_line=(0, 0, 255),
    road_line=(0, 128, 255),
    vehicle=(128, 128, 128))
gpu_ids = range(0, 1)
samples_per_gpu = 4
seed = 0
