# build dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import mmcv
import os

base = 'data/parkinglot/'
work_dir = 'checkpoints/OCR_HRNet_Parkinglot/'
# work_dir = 'checkpoints/parkinglot/'
img_table_file = base+'img_anno.csv'
dataset_name= 'parkinglot'
# model_name = 'OCR_HRNet_Parkinglot'
img_dir = 'images/'
ann_dir = 'labels/'
#load palette
palette = eval(open(base+'color.json', 'r').read())

# @DATASETS.register_module()
# class ParkinglotDataset(CustomDataset):
#   CLASSES = [k for k,v in palette.items()]
#   PALETTE = [v for k,v in palette.items()]
#   def __init__(self, split, **kwargs):
#     super().__init__(img_suffix='', seg_map_suffix='', split=split, **kwargs)
#     assert os.path.exists(self.img_dir) and self.split is not None

# import config
from mmcv import Config
from mmseg.apis import set_random_seed
## Load config
# cfg = Config.fromfile('configs/hrnet/parkinglot.py')# Using HRNetV2
cfg = Config.fromfile('configs/ocrnet/parkinglot_ocr_hrnet.py')# Using OCR+HRNet

# cfg.load_from = 'checkpoints/parkinglot/latest.pth'#'checkpoints/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth'
# cfg.resume_from = 'checkpoints/parkinglot/latest.pth'

cfg.load_from = work_dir+'ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'
cfg.resume_from = ''
# # cfg.pop('resume_from')

# cfg.optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0005)
# cfg.evaluation = dict(interval=10000, metric='mIoU')

# update img_norm
cfg.img_norm_cfg = dict(
    type = 'Normalize',
    mean=[86.2430, 84.9454, 81.4281], 
    std=[43.0072, 42.2812, 42.9458], 
    to_rgb=True)

from mmcv.utils.config import Config, ConfigDict
def update_normalization_params(obj, path='cfg'):
    if isinstance(obj, Config) or isinstance(obj, ConfigDict):
        if 'type' in obj and obj.type == 'Normalize':
            obj.mean = cfg.img_norm_cfg.mean
            obj.std = cfg.img_norm_cfg.std
            print(f'Found {path}:{obj}, updated Normalize params')
            return
        else:
            for k, v in obj.items():
                update_normalization_params(v, path+'.'+k)
    elif isinstance(obj, list): #list
        for obj2 in obj:
            update_normalization_params(obj2, f"{path}[{obj.index(obj2)}]")
    else:
        # print(path, obj)
        pass

update_normalization_params(cfg)


# print(f'Config:\n{cfg.pretty_text}')
with open(work_dir+'config.py', 'w') as f:
    f.write(cfg.pretty_text)

#get gflops for model
# os.system('python tools/get_flops.py configs/hrnet/parkinglot.py --shape 1024 512')

# train and eval
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, init_segmentor, train_segmentor

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)
# model = init_segmentor(cfg, cfg.load_from, device='cuda:0')
# Add an attribute for visualization convenience
model.CLASSES = [k for k,v in palette.items()]
model.PALETTE = [v for k,v in palette.items()]

# Create work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())