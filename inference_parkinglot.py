from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmseg.models import build_segmentor
from mmcv import Config
import mmcv

# config_file = 'configs/hrnet/parkinglot.py'
config_file = 'configs/ocrnet/ocrnet_hr48_parkinglot_config.py'
checkpoint_file = 'checkpoints/iter_160000_final.pth'
CLASSES = ('road', 'curb', 'obstacle', 'chock', 'parking_line', 'road_line', 'vehicle')
PALETTE = [(0, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 128, 255), (128, 128, 128)]


model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = CLASSES
model.PALETTE = PALETTE

# get data
from glob import glob
from random import sample
import pandas as pd
from datetime import datetime
import shutil

base = 'data/parkinglot/'
img_table_file = base+'img_anno.csv'
test_dir = 'output/ocr_parkinglot/'
img_dir = base+'images/'
anno_dir = base+'labels/'
img_table = pd.read_csv(img_table_file)
img_list = img_table.dropna().sample(10).img.to_list()

# infer
def predict_save(model, img:str, out_path=None):
    result = inference_segmentor(model, img)
    if out_path:
        model.show_result(img, result, palette=PALETTE, out_file=out_path)
    else:
        model.show_result(img, result, palette=PALETTE, show=True)

def plot_result(model, img:str):
    result = inference_segmentor(model, img)
    show_result_pyplot(model, img, result, palette=PALETTE)

# img = sample(img_list, 1)[0]

tstr = datetime.now().isoformat()
for path in img_list:
    img_name = path.split('/')[-1]
    anno_name = img_name.split('.')[0]+'.png'
    infer_name = img_name.split('.')[0]+'_predict.jpg'
    img_path = img_dir+img_name
    label_path = anno_dir+anno_name
    out_dir = test_dir + tstr + '/'

    predict_save(model, img_path, out_dir+infer_name)
    shutil.copy2(label_path, out_dir+anno_name)
    shutil.copy2(img_path, out_dir+img_name)

print('Infer finished')