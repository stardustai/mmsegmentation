import glob, json, re, os, shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pylab import mpl
import mmcv
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# import torch.distributed as dist
# dist.init_process_group('gloo', rank=0, world_size=1, init_method='file:///temp/torch_distributed_file')#single GPU
# from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot

CLASSES = ('road', 'curb', 'obstacle', 'chock', 'parking_line', 'road_line', 'vehicle')
PALETTE = [(0, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 128, 255), (128, 128, 128)]
base = 'data/parkinglot/'
img_table_file = base+'img_anno.csv'
dataset_name= 'parkinglot'
img_dir = 'images/'
# ann_dir = 'annotations/'
label_dir = 'labels/'
#load palette
palette = eval(open(base+'color.json', 'r').read())


#-----------convert to P mode ---------------
convert_labels = False
def convertImageToPaletteMode(path, dst=None, palette=None, overwrite=True):
    if not dst:
        file_name = path.split('/')[-1]
        file_name = file_name.replace('.jpg', '.png')
        dst = base+label_dir+file_name
    if not overwrite and os.path.exists(dst):
        try:
            seg_img = Image.open(dst)
            assert seg_img.mode == 'P'
            return
        except Exception as e:
            print(f'Image corrupted with error:', e)
    seg_img = Image.open(path)
    palette = palette or PALETTE
    p = np.array([palette], dtype=np.uint8)
    p_img = Image.fromarray(p).convert('P', palette=Image.ADAPTIVE)
    seg_img1 = seg_img.quantize(colors=len(palette), method=Image.MAXCOVERAGE, palette=p_img)
    seg_arr = np.asarray(seg_img1)
    assert len(np.unique(seg_arr)) <= len(palette)
    seg_img1.save(dst)

if convert_labels == True:
    files = glob.glob(base+'rendered/*.png')
    # for path in tqdm(files):
    #     convertImageToPaletteMode(path)
    with ThreadPoolExecutor(max_workers=30) as executor:
        result = list(tqdm(executor.map(convertImageToPaletteMode, files), total=len(files)))
        assert sum([0 if i is None else 1 for i in result])==0

#------------download image-------------
download_image = False
def download_img(url, path=None):
    name = url.split('/')[-1]
    if not path:
        path = 'data/parkinglot/images/'+name
    if os.path.exists(path):
        pass
        # print(f'already downloaded: {url}')
    content = requests.get(url).content
    if len(content)==0:
        print(f'failed to download file: {url}')
        return url
    with open(path, 'wb') as f:
        f.write(content)

if download_image:
    files = []
    for path in tqdm(glob.glob(base+'csv/*.csv')):
        file = pd.read_csv(path)
        for i, item in file.iterrows():
            res = eval(item.resource)
            assert len(res)==1
            res = res[0]
            url = res['image_source']['url']
            files.append(url)
    print(f'Downloadingn {len(files)} images')
    with ThreadPoolExecutor(max_workers=20) as executor:
        result = list(tqdm(executor.map(download_img, files), total=len(files)))

    # for url in tqdm(files):
    #     result.append(download_img(url))
    result_ = []
    for u in result:
        if u:
            result_.append(u)
    result = result_
    if len(result)>0:
        with open('data/parkinglot/failed.txt', 'w') as f:
            for p in [p for p in result if p is not None]:
                f.write(p)
        print(f'Images downloaded with {len(result)} errors')

#-----------remove timestamp on img-----------
#2016-01-01-00-57-20-00054_bird_ml_1593591652.jpg
remove_ts = False
renamed_img = []
if remove_ts:
    for path in tqdm(glob.glob(base+img_dir+'*.jpg')):
        path1 = re.sub(r'_\d{10}(?=\.jpg)', '', path)
        if path1 != path:
            os.rename(path, path1)
            renamed_img.append(path)
        



UPDATE_TABLE = False
if os.path.exists(img_table_file) and not UPDATE_TABLE:
    img_table = pd.read_csv(img_table_file)
else:
    images = []
    # scan images
    for path in tqdm(glob.glob(base+'csv/*.csv')):
        file = pd.read_csv(path)
        """
        task_id,resource,answer,metadata
        ,"[{'image_source': {'url': 'https://stardust-data.oss-cn-hangzhou.aliyuncs.com/projects/%E4%B8%8A%E6%B1%BD%E5%81%9C%E8%BD%A6%E4%BD%8D%E6%95%B0%E6%8D%AE/%E5%88%86%E5%89%B2/4.22_annotation/2020-04-22-14-08-29-00003_bird_ml.jpg', 'meta': {'width': 1280, 'height': 1280}}}]",,
        """
        for i, item in file.iterrows():
            res = eval(item.resource)
            assert len(res)==1
            res = res[0]
            url = res['image_source']['url']
            meta = res['image_source']['meta']
            # batch = url.split('/')[-2]
            # batch = re.findall(r'^\d*\.\d*', batch)[0]
            name = url.split('/')[-1]
            name = re.sub(r'_\d{10}(?=\.jpg)', '', name)
            images.append({
                # 'batch': str(batch),
                'img': name,
                'url': url,
                'meta': meta
            })
    img_table = pd.DataFrame(images)

    #scan annotated image
    unmatched_render = []
    for path in tqdm(glob.glob(base+'labels/*.png', recursive=True)):
        file_name = path.split('/')[-1]
        file_name_jpg = file_name.replace('.png', '.jpg')
        # batch = path.split('/')[-2]
        # batch = re.findall(r'\d{1,2}\.\d{1,2}', path)[0]
        row = img_table.query('img==@file_name_jpg')
        if len(row) == 1:
            # file_name = file_name.replace('.jpg', '.png')
            img_table.loc[row.index, 'annotated_img'] = file_name
            # tgt = base+ann_dir+file_name
            # if not os.path.exists(tgt):
            #     convertImageToPaletteMode(path, tgt) #后面批量处理
            #     shutil.copyfile(path, tgt)
        elif len(row)>1:
            print(f'Duplicated image: {path}')
        else:
            unmatched_render.append({'annotated_img': file_name})
    img_table = img_table.append(pd.DataFrame(unmatched_render), ignore_index=True)
    print(f'Found {img_table.annotated_img.dropna().shape[0]}  matched render and {len(unmatched_render)} unmatched')
    pd.Series(unmatched_render).to_csv(base+'rendered/unmatched.txt', index=False, header=False)

    #check download image
    for path in tqdm(img_table.img.dropna()):
        assert os.path.exists(base+'images/'+path)

    img_table.to_csv(img_table_file)

# check files
check_bad_images = False
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
if check_bad_images:
    #jpg
    for path in tqdm(glob.glob(base+'images/*.jpg')):
        try:
            image = Image.open(path).convert("RGB")
            img_arr = np.asarray(image, dtype=np.float32)
            assert img_arr.dtype == np.float32
        except Exception as e:
            print(e, path)
            name = path.split('/')[-1]
            url = img_table.query('img==@name').url.values[0]
            download_img(url)
            print(f'Downloaded image: {name}')



# stats
n_record = img_table.img.dropna().shape[0]
n_labeled = img_table.dropna().shape[0]
print(f'Annotated img: {n_labeled}/{n_record}({n_labeled/n_record*100:.2f}%)')


# show generated image
# import matplotlib.patches as mpatches
# img = Image.open(base+label_dir+img_table.dropna().sample(1).annotated_img.iloc[0])
# plt.figure(figsize=(8, 6))
# im = plt.imshow(np.array(img.convert('RGB')))
# # create a patch (proxy artist) for every color 
# patches = [mpatches.Patch(color=np.array(v)/255., label=k) for k,v in palette.items()]
# # put those patched as legend-handles into the legend
# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')
# plt.show()

# make train val split file
train_val_set = img_table.dropna().sample(frac=1)
train_val_set = train_val_set.img.apply(lambda x:x.split('.')[0])
length = train_val_set.shape[0]
n_train = int(length*0.95)
train_val_set[:n_train].to_csv(base+'train.csv', index=False, header=False)
train_val_set[n_train:].to_csv(base+'val.csv', index=False, header=False)
print('split file updated')

# Test image


# calculate normalization param: mean and std
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import Tensor
calculate_mean_variance = False
class MyDataset(Dataset):
    def __init__(self, image_list):
        assert isinstance(image_list, list)
        self.image_list = image_list
    
    def __getitem__(self, index):
        path = base+img_dir+self.image_list[index]
        image = Image.open(path).convert("RGB")
        img_arr = np.asarray(image, dtype=np.float32)
        assert img_arr.dtype == np.float32
        return img_arr

    def __len__(self):
        return len(self.image_list)


if calculate_mean_variance:
    dataset = MyDataset(img_table.dropna().img.to_list())
    # dataset = datasets.ImageFolder(base+img_dir)# Only works in specified folder structure!
    loader = DataLoader(
        dataset,
        batch_size=100,
        num_workers=1,
        shuffle=False
    )

    mean = Tensor([0,0,0])
    std = Tensor([0,0,0])
    n_samples= 0
    for data in tqdm(loader):
        batch_samples = data.size(0)
        data2 = data.view(-1, 3)
        mean += data2.mean(0)
        std += data2.std(0)
        n_samples += 1

    mean /= n_samples #[86.2430, 84.9454, 81.4281]
    std /= n_samples #[43.0072, 42.2812, 42.9458]
    print(f'mena:{mean}')
    print(f'std:{std}')


# calculate class weight
calculate_class_weight = True
import math
def getClassWeight(name):
    path = 'data/parkinglot/labels/'+name
    image = Image.open(path)
    img_arr = np.asarray(image)
    unique, counts = np.unique(img_arr.flatten(), return_counts=True)
    classes = [CLASSES[i] for i in unique]
    value_counts = dict(zip(classes, counts))
    return value_counts

labels = img_table.dropna().annotated_img.to_list()
# results = []
# for name in tqdm(labels):
#     results.append(getClassWeight(name))

with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(getClassWeight, labels), total=len(labels)))

class_counts = {k:0 for k in CLASSES}
for value_counts in results:
    for k, v in value_counts.items():
        class_counts[k] += v
class_counts = pd.Series(class_counts)
class_weights = class_counts.mean()/class_counts
class_weights = class_weights.apply(lambda x:math.sqrt(x))
print(f'Class weights are:{class_weights}')
'''
road            2.931730
curb            0.658505
obstacle             inf
chock                inf
parking_line    6.421162
road_line       2.968377
vehicle         0.474590
'''
