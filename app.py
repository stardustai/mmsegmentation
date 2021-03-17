from numpy.lib.polynomial import poly
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from PIL import Image, ImagePalette
import glob, os, random, sys
import numpy as np
import pandas as pd
from skimage import measure
from PIL import Image, ImageDraw, ImageFont
from imantics import Polygons, Mask, Annotation
# import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from service_streamer import ThreadedStreamer, Streamer
import cv2
from datetime import datetime
from collections import defaultdict
from tools.topo import extractPolygons, snapPolygonPoints, get_polygon_dict


base = 'data/parkinglot/'
palette = eval(open(base+'color.json', 'r').read())
CLASSES = ('road', 'curb', 'obstacle', 'chock', 'parking_line', 'road_line', 'vehicle')
PALETTE = [(0, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 128, 255), (128, 128, 128)]
config_file = 'configs/ocrnet/ocrnet_hr48_parkinglot_config.py'
checkpoint_file = 'checkpoints/iter_160000_final.pth'

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 10 # 10MB max
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']


@app.route("/seg_parkinglot", methods=["POST"])
def main():
    try:
        file = request.files['image']
        assert file.content_type[:6] == 'image/', 'must upload image type file'
        filename = secure_filename(file.filename)
        uploaded_img = file.read()
    except Exception as e:
        return str(e)
    img_path = processImg(file, filename)
    outputs = streamer.predict([img_path])
    view_img = request.form.get('view_img', False)
    view_img = True if view_img == 'true' else False
    if view_img:# == 'true' or view_img is True:
        ext = img_path.split('.')[-1]
        output_path = img_path.replace(ext, 'result.'+ext)
        model.show_result(img_path, outputs, palette=PALETTE, out_file=output_path)
        #return img0
        return send_file(output_path)
    tolerance = int(request.form.get('tolerance', 2))
    results = polygonize(outputs, tolerance=tolerance)
    return jsonify(results)

def processImg(uploaded_file, filename):
    datestr = datetime.now().isoformat().split('T')[0]
    tmp_path = 'output/'+datestr
    os.makedirs(tmp_path, exist_ok=True)
    tmp_path += '/'+filename
    # img = cv2.imdecode(np.frombuffer(uoloaded_file, np.uint8), cv2.IMREAD_UNCHANGED)
    # assert img is not None, 'Not Image'
    # cv2.imwrite(tmp_path, img)
    img = Image.open(uploaded_file.stream)
    img.save(tmp_path)
    return tmp_path

# infer and plot
def predict(img:str):
    if type(img) is list:
        img = img[0]
    result = inference_segmentor(model, img)
    return result

def polygonize(result, tolerance=2, draw_img=None):
    polygon_data = extractPolygons(result)
    #sort by area from large to small
    polygons_data.sort(key=lambda p:p['area'], reverse=True)
    # join vertex of polygons
    snapPolygonPoints(polygons_data, result)#snap points

    #simplify polygons using topo
    topo_data = [Feature(
                geometry = Polygon([p['polygon']]),
                properties = {"name": p['label']}
                ) for p in polygons_data]
    fc = FeatureCollection(topo_data)
    topo = tp.Topology(fc, prequantize=True, topology=True, shared_coords=True)
    topo_s = topo.toposimplify(
        epsilon=tolerance, 
        simplify_algorithm='dp', 
        )
    # convert to desired structure 
    polygons_strctured = get_polygon_dict(topo_s)
    return polygons_strctured


def get_random_img(n=1):
    img_table_file = base+'img_anno.csv'
    # test_dir = 'output/ocr_parkinglot/'
    img_dir = base+'images/'
    # anno_dir = base+'labels/'
    img_table = pd.read_csv(img_table_file)
    paths = img_table.dropna().sample(n).img.to_list()
    # img_name = path.split('/')[-1]
    # img_path = img_dir+img_name
    return paths

def getBoxAreaAndCenter(points):
    xmax = points[:,0].max()
    xmin = points[:,0].min()
    ymax = points[:,1].max()
    ymin = points[:,1].min()
    area = (xmax-xmin)*(ymax-ymin)
    center = ((xmax+xmin)/2,(ymax+ymin)/2)
    return area, center

    # area = Annotation(polygons=points.tolist(), width=points[:,0].max(), height=points[:,1].max()).area
    size = (points[:,0].max(), points[:,1].max())
    mask = np.zeros(size, np.uint8)
    mask = cv2.fillPoly(mask, [points], 1)
    area = mask.sum()
    center = ((mask*range(1,size[0]+1)).mean(), (mask*range(1,size[1]+1)).mean())
    return area, center

## Local test
# paths = get_random_img()
# img_path = paths[0]
# result = predict_save(model, img_path)
# polygons = polygonize(result, draw_img=img_path)


if __name__ == "__main__":
    # start child thread as worker
    streamer = ThreadedStreamer(predict, batch_size=4, max_latency=0.5)
    # spawn child process as worker for multiple GPU
    # streamer = Streamer(predict_save, batch_size=4, max_latency=0.1)
    app.run(port=5005, debug=False, host= '0.0.0.0')
    print('Flask started')