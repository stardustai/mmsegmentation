_base_ = './ocrnet_hr18_512x1024_160k_cityscapes.py'
work_dir = 'checkpoints/OCR_HRNet_Parkinglot/'
dataset_type = 'ParkinglotDataset'
# load_from = work_dir+'latest.pth'
resume_from = work_dir+'latest.pth'
base = 'data/parkinglot/'
palette = eval(open(base+'color.json', 'r').read())

checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=5)
evaluation = dict(interval=10000, metric='mIoU')
gpu_ids = range(4)
# gpu_ids = range(1)
if len(gpu_ids)>1:
    norm_cfg = dict(type='SyncBN', requires_grad=True)
    print('Found multiple GPU, SyncBN enabled!')
else:
    norm_cfg = dict(type='BN', requires_grad=True)

# enable fp16
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)

seed = 0
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        norm_cfg = norm_cfg,
        with_cp=True,
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            num_classes = len(palette),
            type='FCNHead',
            # fp16_enabled = True,
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            # num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            num_classes = len(palette),
            type='OCRHead',
            # fp16_enabled = True,
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            # num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])


img_norm_cfg = dict(
    mean=[86.2430, 84.9454, 81.4281], 
    std=[43.0072, 42.2812, 42.9458], 
    to_rgb=True)
crop_size = (1800, 1800)
data = dict(
    samples_per_gpu = 8,
    workers_per_gpu= 8,
    data_root = base,
    train = dict(
        type = dataset_type,
        data_root = base,
        img_dir = 'images',
        ann_dir = 'labels',
        split = 'train.csv'
    ),
    val = dict(
        type = dataset_type,
        data_root = base,
        img_dir = 'images',
        ann_dir = 'labels',
        split = 'val.csv'
    ),
    test = dict(
        type = dataset_type,
        data_root = base,
        img_dir = 'images',
        ann_dir = 'labels',
        split = 'val.csv'
    )
)