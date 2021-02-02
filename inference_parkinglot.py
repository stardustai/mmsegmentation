from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmseg.models import build_segmentor
from mmcv import Config

config_file = 'configs/hrnet/fcn_hr48_512x1024_160k_parkinglot.py'
checkpoint_file = 'checkpoints/parkinglot/iter_70000.pth'
CLASSES = ('road', 'curb', 'obstacle', 'chock', 'parking_line', 'road_line', 'vehicle')
PALETTE = [(0, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 128, 255), (128, 128, 128)]

# build the model from a config file and a checkpoint file
# cfg = Config.fromfile('configs/hrnet/fcn_hr48_512x1024_160k_parkinglot.py')
# cfg.resume_from = checkpoint_file
# model = build_segmentor(cfg.model)

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = CLASSES
model.PALETTE = PALETTE

# get data
from glob import glob
from random import sample
img_list = glob('data/parkinglot/images/*.jpg')
img = sample(img_list, 1)[0]

# infer
result = inference_segmentor(model, img)
show_result_pyplot(model, img, result, palette=PALETTE)