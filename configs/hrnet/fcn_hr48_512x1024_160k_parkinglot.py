_base_ = './fcn_hr18_512x1024_160k_cityscapes.py'
dataset_type = 'ParkinglotDataset'
load_from = 'checkpoints/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth'
work_dir = 'checkpoints/parkinglot/'
seed = 0
gpu_ids = range(1)
base = 'data/parkinglot/'
palette = eval(open(base+'color.json', 'r').read())


gpu_ids = range(1)
checkpoint_config = dict(by_epoch=False, interval=10000)

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    # CLASSES = [k for k,v in palette.items()],
    # PALETTE = [v for k,v in palette.items()],
    backbone=dict(
        norm_cfg = dict(type='BN', requires_grad=True),
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        num_classes = len(palette),
        norm_cfg = dict(type='BN', requires_grad=True),
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