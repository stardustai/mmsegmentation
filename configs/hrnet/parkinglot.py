_base_ = './fcn_hr18_512x1024_160k_cityscapes.py'
dataset_type = 'ParkinglotDataset'
load_from = 'checkpoints/parkinglot/latest.pth'
resume_from = 'checkpoints/parkinglot/latest.pth'
work_dir = 'checkpoints/parkinglot/'
base = 'data/parkinglot/'
palette = eval(open(base+'color.json', 'r').read())
img_norm_cfg = dict(
    mean=[86.2430, 84.9454, 81.4281], 
    std=[43.0072, 42.2812, 42.9458], 
    to_rgb=True)

seed = 0
gpu_ids = range(1)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=5)
# checkpoint_config = dict(interval=1, by_epoch=True)   # 每1个epoch存储一次模型
evaluation = dict(interval=10000, metric='mIoU')

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    # CLASSES = [k for k,v in palette.items()],
    # PALETTE = [v for k,v in palette.items()],
    backbone=dict(
        norm_cfg = norm_cfg,
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        num_classes = len(palette),
        norm_cfg = norm_cfg,
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

data = dict(
    samples_per_gpu = 2,
    workers_per_gpu= 2,
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