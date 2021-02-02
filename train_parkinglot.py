# build dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import mmcv
import os

base = 'data/parkinglot/'
img_table_file = base+'img_anno.csv'
dataset_name= 'parkinglot'
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
cfg = Config.fromfile('configs/hrnet/fcn_hr48_512x1024_160k_parkinglot.py')
# cfg.dataset_type = 'ParkinglotDataset'
# cfg.gpu_ids = range(1)
# cfg.checkpoint_config = dict(by_epoch=False, interval=10000)

# # data
# cfg.data.samples_per_gpu = 2
# cfg.data.workers_per_gpu= 2
# cfg.data_root = base
# cfg.data.train.type = cfg.dataset_type
# cfg.data.train.data_root = cfg.data_root
# cfg.data.train.img_dir = img_dir
# cfg.data.train.ann_dir = ann_dir
# cfg.data.train.split = 'train.csv'

# cfg.data.val.type = cfg.dataset_type
# cfg.data.val.data_root = cfg.data_root
# cfg.data.val.img_dir = img_dir
# cfg.data.val.ann_dir = ann_dir
# cfg.data.val.split = 'val.csv'

# cfg.data.test.type = cfg.dataset_type
# cfg.data.test.data_root = cfg.data_root
# cfg.data.test.img_dir = img_dir
# cfg.data.test.ann_dir = ann_dir
# cfg.data.test.split = 'val.csv'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'checkpoints/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth'
cfg.resume_from = 'checkpoints/parkinglot/iter_70000.pth'

# Set up working dir to save files and logs.
# cfg.work_dir = 'checkpoints/'+dataset_name
# train config
# cfg.seed = 0
# set_random_seed(0, deterministic=False)
# cfg.model.decode_head.num_classes = len(palette)
# cfg.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.model.backbone.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)

print(f'Config:\n{cfg.pretty_text}')

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