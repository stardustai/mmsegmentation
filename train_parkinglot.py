# build dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import mmcv
import os, time
import torch
from mmseg.utils import collect_env, get_root_logger
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.runner import init_dist
from mmseg import __version__
# from mmcv import Config
from mmseg.apis import set_random_seed
from mmcv.utils.config import Config as ConfigRoot, ConfigDict

base = 'data/parkinglot/'
work_dir = 'checkpoints/Parkinglot_ocr_hr_norm_cw/'
# img_table_file = base+'img_anno.csv'
# dataset_name= 'parkinglot'
# model_name = 'OCR_HRNet_Parkinglot'
# img_dir = 'images/'
# ann_dir = 'labels/'
#load palette
# palette = eval(open(base+'color.json', 'r').read())
# set cudnn_benchmark
torch.backends.cudnn.benchmark = True

# import config
# cfg = Config.fromfile('configs/hrnet/parkinglot.py')# Using HRNetV2
cfg = Config.fromfile('configs/ocrnet/ocrnet_hr48_parkinglot_config.py')# Using OCR+HRNet
cfg.work_dir = work_dir
# cfg.load_from = 'checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'
cfg.load_from = 'checkpoints/OCR_HRNet_Parkinglot_P100/latest.pth'
# cfg.resume_from = 'checkpoints/Parkinglot_ocr_hr_norm_cw/latest.pth'
cfg.runner = dict(type='IterBasedRunner', max_iters=640000)

#GPU
# n_gpu = torch.cuda.device_count()
# print(f'Found {n_gpu} GPUs')
# cfg.gpu_ids = range(n_gpu)
# if n_gpu>1:
#     cfg.norm_cfg = dict(type='SyncBN', requires_grad=True)
#     cfg.samples_per_gpu = 8
#     print('Found multiple GPU, SyncBN enabled!')
# else:
#     cfg.norm_cfg = dict(type='BN', requires_grad=True)
#     cfg.samples_per_gpu = 4
#     print('Using single GPU with batch size 4')

# adjust learning rate
cfg.optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# In MMSegmentation, you may add following lines to config to make the LR of heads 10 times of backbone.
# cfg.optimizer=dict(
#     type='AdamW', lr=0.0001, weight_decay=0.00001,
    # type='SGD', lr=0.0003, momentum=0.9, weight_decay=0.00001,
    # paramwise_cfg = dict(
    #     custom_keys={
    #         'head': dict(lr_mult=2)})
# )


def update_config(obj, path='cfg'):
    if isinstance(obj, ConfigRoot) or isinstance(obj, ConfigDict or isinstance(obj, dict)):
        check_children = False
        if 'type' in obj:
            if obj.type == 'Normalize':
                obj.mean = cfg.img_norm_cfg.mean
                obj.std = cfg.img_norm_cfg.std
                print(f'Updated `Nomalize` at {path} -> {obj}')
            elif obj.type == 'Resize':
                obj.img_scale=(1800, 1800)
                print(f'updated `Resize` at {path} -> {obj}')
            # elif obj.type == 'RandomCrop':
            #     obj.crop_size=(1024, 1024)
            #     print(f'updated `RandomCrop` at {path}')
            # elif obj.type == 'Pad':
            #     obj.size=(1024, 1024)
            #     print(f'updated `Pad` at {path}')
            # elif obj.type == 'MultiScaleFlipAug':
                obj.img_scale=(1800, 1800)
                print(f'updated `MultiScaleFlipAug` at {path} -> {obj}')
            else:
                check_children = True
        else:
            check_children = True
        if check_children:
            for k, v in obj.items():
                update_config(v, path+'.'+k)
    elif isinstance(obj, list): #list
        for obj2 in obj:
            update_config(obj2, f"{path}[{obj.index(obj2)}]")
    else:
        if type(obj) not in [str, tuple, int, bool, float, range]:
            print(path, obj)
        pass

update_config(cfg)

# init distributed env first, since logger depends on the dist info.
distributed = True
if distributed == True:
    init_dist('pytorch', **cfg.dist_params)
# init the logger before other steps
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
# init the meta dict to record some important information such as
# environment info and seed, which will be logged
meta = dict()
# log env info
env_info_dict = collect_env()
env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
meta['env_info'] = env_info
# log some basic info
logger.info(f'Distributed training: {distributed}')
# set random seeds
seed = 0
if seed is not None:
    logger.info(f'Set random seed to {seed}, deterministic: ' f'{True}')
    set_random_seed(seed, deterministic=True)
cfg.seed = seed
meta['seed'] = seed

# get gflops for model
# os.system('python tools/get_flops.py configs/hrnet/parkinglot.py --shape 1024 512')

# train and eval
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, init_segmentor, train_segmentor

# Build the dataset
datasets = [build_dataset(cfg.data.train)]
if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    datasets.append(build_dataset(val_dataset))
if cfg.checkpoint_config is not None:
    # save mmseg version, config file content and class names in
    # checkpoints as meta data
    cfg.checkpoint_config.meta = dict(
        mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
        config=cfg.pretty_text,
        CLASSES=datasets[0].CLASSES,
        PALETTE=datasets[0].PALETTE)

# save config
# cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
# with open(work_dir+'config.py', 'w') as f:
#     f.write(cfg.pretty_text)

# log model info
model = build_segmentor(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
# logger.info(model)
# model = init_segmentor(cfg, cfg.load_from, device='cuda:0')
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
model.PALETTE = datasets[0].PALETTE

# Create work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=distributed, validate=True, timestamp=timestamp, meta=meta)