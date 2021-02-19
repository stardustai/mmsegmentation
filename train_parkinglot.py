# build dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import mmcv
import os

base = 'data/parkinglot/'
work_dir = 'checkpoints/Parkinglot_ocr_hr_norm_cw/'
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
cfg = Config.fromfile('configs/ocrnet/ocrnet_hr48_parkinglot_config.py')# Using OCR+HRNet
cfg.work_dir = work_dir
cfg.load_from = 'checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'
# cfg.load_from = 'checkpoints/OCR_HRNet_Parkinglot/iter_160000.pth'
cfg.resume_from = 'checkpoints/Parkinglot_ocr_hr_norm_cw/latest.pth'
cfg.runner = dict(type='IterBasedRunner', max_iters=640000)

# adjust learning rate
# cfg.optimizer = dict(type='SGD', lr=3e-4, momentum=0.9, weight_decay=0.0001)
# In MMSegmentation, you may add following lines to config to make the LR of heads 10 times of backbone.
cfg.optimizer=dict(
    # type='AdamW', lr=0.001, weight_decay=0.0001,
    type='SGD', lr=0.0003, momentum=0.9, weight_decay=0.00001,
    paramwise_cfg = dict(
        custom_keys={
            'head': dict(lr_mult=2)})
)




from mmcv.utils.config import Config, ConfigDict
def update_config(obj, path='cfg'):
    if isinstance(obj, Config) or isinstance(obj, ConfigDict or isinstance(obj, dict)):
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


# print(f'Config:\n{cfg.pretty_text}')
with open(work_dir+'config.py', 'w') as f:
    f.write(cfg.pretty_text)

# get gflops for model
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