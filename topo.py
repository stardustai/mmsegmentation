from math import radians
from numpy.lib.arraysetops import isin
from numpy.lib.polynomial import poly
import topojson as tp
import pickle, random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from imantics import Polygons, Mask, Annotation
from geojson import Feature, Polygon, FeatureCollection, LineString, MultiLineString, MultiPolygon
from skimage import measure
from skimage.feature import canny
import skimage.morphology as sm
from collections import defaultdict
from tqdm import tqdm
# from skimage import io,color

palette = eval(open('data/color.json', 'r').read())
CLASSES = ('road', 'curb', 'obstacle', 'chock', 'parking_line', 'road_line', 'vehicle')
PALETTE = [(0, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 128, 255), (128, 128, 128)]
tolerance = 1.5
draw_img = True
img_path = 'test.jpg'

def getBoxAreaAndCenter(points):
    center = (points[:,0].mean(), points[:,1].mean())
    area = cv2.contourArea(points)
    return area, center


# load from topo_data
# data = pickle.load(open('topo_data', 'rb')) 

# load from mask
polygons_data = []
img = Image.open('mask.png')
result = np.asarray(img)
# polygons = cv2.findContours(np.where(result==1, 1, 0)\
# .astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] #can't work on single image

## The morphological opening on an image is defined as an erosion followed by a dilation. 
# Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks. 
# This tends to “open” up (dark) gaps between (bright) features.
# result = sm.opening(result, sm.disk(2))  #用边长为2的圆形滤波器进行膨胀滤波
# result = sm.dilation(result,sm.disk(tolerance))  #用边长为5的正方形滤波器进行膨胀滤波
# result = sm.closing(result, sm.disk(3))

## find edge (NOT WORKING)
# edge = canny(result,1,0.5,1)
# edge = sm.closing(edge, sm.disk(3))
# edge = sm.skeletonize(edge)
# polygons = measure.find_contours(edge)
# polygons = []
# for polygon in polygons:
#     polygon = measure.approximate_polygon(polygon, 1)
#     polygon[:,0], polygon[:,1] = polygon[:,1], polygon[:,0]
#     # polygon = [(i[1], i[0]) for i in polygon]
#     polygons.append([polygon.tolist()])

# mp = MultiPolygon(polygons)
# topo = tp.Topology(mp, prequantize=4e3, topology=True, shared_coords=False)
# topo_s = topo.toposimplify(
#     epsilon=tolerance, 
#     simplify_algorithm='vw', 
#     simplify_with='simplification', 
#     )

# find polygon
for label in palette:
    i = CLASSES.index(label)
    mask = np.where(result==i, 1, 0).astype(np.uint8)
    # polygons = Mask(mask).polygons().points # using imantics
    # polygons = measure.find_contours(mask, 0.8) # using skimage
    # using cv2
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE, #cv2.CHAIN_APPROX_TC89_L1, 
        offset=(-1, -1)
    )[0]
    # polygons = [cv2.convexHull(p) for p in polygons] #convexHull, will not find outer edge
    polygons = [p.squeeze() for p in polygons]
    print(f'{label}:{len(polygons)}')
    for j, polygon in enumerate(polygons):
        if polygon.shape[0] <= 2:
            continue
        # if (polygon[0] != polygon[-1]).any():
            # print('---> Polygon not closed:', polygon)
            # polygon = np.append(polygon, np.expand_dims(polygon[0],0), axis=0)
            # assert (polygon[0]==0).any() and (polygon[0]==0).any(), f'---> polygon not closed and not on wall:{polygon}'
            # if polygon[0,0] == polygon[-1,0] or polygon[0,1] == polygon[-1,1]:
            #     #add one point to close polygon
            #     polygon = np.append(polygon, np.expand_dims(polygon[0],0), axis=0)
            # else:
            #     # add two points
            #     polygon = np.append(polygon, [0,0,polygon[-1,0],polygon[-1,1]]).reshape((-1,2))
        # polygon[:,0], polygon[:,1] = polygon[:,1], polygon[:,1] #reverse for skimage
        area, center = getBoxAreaAndCenter(polygon)
        polygon_feat = Feature(
            geometry=Polygon([polygon.tolist()]),
            # geometry=LineString(polygon.tolist()),
            properties={"name":label}
        )
        polygon_approximated = measure.approximate_polygon(polygon, tolerance)
        label2 = label+str(j)
        # print(f'{label2} -> center:{center} area:{area} points:{len(polygon)}->{len(polygon_approximated)}')
        if area >= 100:
            polygons_data.append({
                'label': label,
                'polygon': polygon.tolist(),
                'topo': polygon_feat,
                'approximated': polygon_approximated,
                'area': area,
                'center': center,
                'label2': label2
            })
        else:
            print(f'---> Small region {label} with area:{area} and length:{polygon.shape[0]}')
    
from math import sin, cos, atan, pi
def snapPolygonPoints(polygons_data:list, mask:np.ndarray):
    # create a lookup table to register polygon vertices
    # with value on [x, y, 1]-> i'th polygon and 
    # with value on [x, y, 2]-> j'th coordinates of i'th polygon
    vertices_lookup = np.full(list(mask.shape)+[2], fill_value=-1) 
    polygons = [i['polygon'] for i in polygons_data]
    labels = [i['label'] for i in polygons_data]
    label_indices = [CLASSES.index(l) for l in labels]

    # utils
    radius = 1
    x_upper, y_upper = mask.shape
    _validate_coord = lambda x,y: x >= 0 and y >= 0 and x < x_upper and y < y_upper
    _lookup = lambda x,y: vertices_lookup[x, y, 0] if _validate_coord(x,y) else None 
    _lookup_check_pair = lambda x,y,label: _lookup(x, y) != -1 and _lookup(x, y) != label
    # _lookup_range = lambda x0, x1, y0, y1: np.array([[_lookup(x, y) for x in range(x0,x1)] for y in range(y0, y1)])
    _lookup_scope = lambda x, y: np.array([[_lookup(x, y) for x in range(x-3, x+4)] for y in range(y-3, y+4)])
    _lookup_mask = lambda x,y: mask[y,x] if _validate_coord(x,y) else None # need to reverse x,y from OpenCV to NumPy
    _mask_scope = lambda x,y: np.array([[_lookup_mask(x_,y_) for x_ in range(x-3, x+4)] for y_ in range(y-3, y+4)])

    # create lookup table
    for i, polygon in enumerate(polygons):
        for j, (x, y) in enumerate(polygon):
            assert x == round(x) and y == round(y)
            assert _validate_coord(x, y)
            assert _lookup(x, y) == -1
            vertices_lookup[x, y, :] = [i,j]

    points_snaped = []
    points_missed = []
    # search for vertices to merge
    for i, polygon in tqdm(enumerate(polygons)):
        label_index = label_indices[i]
        for j, (x, y) in enumerate(polygon):
            if type(x) is float or type(y) is float or _lookup(x, y) == -1:
                continue #snapped
            assert _lookup(x, y) == i
            x1, y1 = polygon[j-1]#previous point
            x2, y2 = polygon[(j+1)%len(polygon)]#next point
            degree = atan((y2-y1)/((x2-x1)+1e-8)) # degree of border line
            d1, d2 = degree+pi/2, degree-pi/2 # +-90° to border line
            # find two candidate coordinates, one inside and the other outside of contour
            x_target1, y_target1 = round(x + radius * cos(d1)), round(y + radius * sin(d1))#target1 coords
            x_target2, y_target2 = round(x + radius * cos(d2)), round(y + radius * sin(d2))#target2 coords
            x_up, y_up = x, y-1
            x_down, y_down = x, y+1
            x_left, y_left = x-1, y
            x_right, y_right = x+1, y
            # get target coordinate
            # check whether mask is outside and
            # check whether candidate coordinates next another polygon's vertex
            if _lookup_mask(x_target1, y_target1) != label_index and _lookup_check_pair(x_target1, y_target1, label_index):
                x_t, y_t = x_target2, y_target2
            elif _lookup_mask(x_target2, y_target2) != label_index and _lookup_check_pair(x_target2, y_target2, label_index):
                x_t, y_t = x_target1, y_target1
            elif _lookup_mask(x_up, y_up) != label_index and _lookup_check_pair(x_up, y_up, label_index):
                x_t, y_t = x_up, y_up
            elif _lookup_mask(x_down, y_down) != label_index and _lookup_check_pair(x_down, y_down, label_index):
                x_t, y_t = x_down, y_down
            elif _lookup_mask(x_left, y_left) != label_index and _lookup_check_pair(x_left, y_left, label_index):
                x_t, y_t = x_left, y_left
            elif _lookup_mask(x_right, y_right) != label_index and _lookup_check_pair(x_right, y_right, label_index):
                x_t, y_t = x_right, y_right
            else:
                print(f'No adjacent vertice found at [{i}]({x},{y}):\n{_lookup_scope(x, y)}')
                points_missed.append((i, j, (x, y)))
                continue
            #snap points
            j_t, n_t = vertices_lookup[x_t, y_t, :] #find j'th polygon and n'th coordinates
            print(f'Found adjacent vertice [{i}]({x},{y})<->[{j_t}]({x_t},{y_t})')
            x_c, y_c = polygons[j_t][n_t] # get coordinates
            x_a, y_a = (x+x_c)/2, (y+y_c)/2 # calculate average
            polygon[j] = [x_a, y_a]
            polygons[j_t][n_t] = [x_a, y_a]
            # remove from lookup table
            vertices_lookup[x, y, 1] = -1
            vertices_lookup[x_t, y_t, 1] = -1
            points_snaped.append((i, j, (x, y)))
            points_snaped.append((j_t, n_t, (x, y)))
    #make sure all points are cleared
    assert (vertices_lookup[:,:,1]!=-1).sum()==0




#simplify polygons using topo
polygons_data.sort(key=lambda p:p['area'], reverse=True)#sort by area
snapPolygonPoints(polygons_data, result)#snap points
topo_data = [i['topo'] for i in polygons_data]
pickle.dump(topo_data, open('topo_data', 'wb'))
fc = FeatureCollection(topo_data)
topo = tp.Topology(fc, prequantize=True, topology=True, shared_coords=False)
topo_s = topo.toposimplify(
    epsilon=tolerance, 
    simplify_algorithm='vw', 
    simplify_with='simplification', 
    )

with open('topo.svg', 'w') as f:
    f.write(topo_s.to_svg())
with open('topo0.svg', 'w') as f:
    f.write(topo.to_svg())

def get_polygon_dict(topo):
    polygons_strctured = defaultdict(list)
    topo_json = eval(topo.to_geojson())
    for geo in topo_json['features']:
        label = geo['properties']['name']
        polygon = geo['geometry']['coordinates'][0]
        polygons_strctured[label].append(polygon)
    return polygons_strctured

polygons_strctured = get_polygon_dict(topo_s)
polygons_strctured_0 = get_polygon_dict(topo)

# draw image for debugging
def draw_polygon(label, polygon, draw):
    p = [tuple(i) for i in polygon]
    polygon = np.array(polygon)
    center = [polygon[:,0].mean(), polygon[:,1].mean()]
    center[0] -= 5*len(label)
    center[1] -= 10
    color = PALETTE[CLASSES.index(label)]
    color2 = tuple(list(color)+[128]) #transparent
    color3 = tuple([200-c for c in color]) #invert
    # draw.point(polygon3, fill=color3)
    # draw.line(polygon3, width=1, fill=color, joint='curve')
    # draw.polygon(polygon3, fill=color2, outline=color)
    draw.polygon(p, fill=color2, outline=color)
    draw.point(p, fill=color3)
    # draw.text(center, label, fill=color3, font=fnt,)

if draw_img:
    im = Image.open(img_path)
    draw = ImageDraw.Draw(im, mode='RGBA')
    im2 = Image.open(img_path)
    draw2 = ImageDraw.Draw(im2, mode='RGBA')
    fnt = ImageFont.truetype("Arial", 20)
    # draw approximated polygon (inferior)
    for data in polygons_data:
        label = data['label']
        label2 = data['label2']
        polygon1 = data['polygon']
        polygon2 = data['approximated']
        draw_polygon(label, polygon1, draw)
        draw_polygon(label, polygon2, draw2)
    im.save('seg_result1.png') #mask -> polygon
    im2.save('seg_result2.png') #mask -> approximate polygon

    # draw topo graph
    img3 = Image.open(img_path)
    draw3 = ImageDraw.Draw(img3, mode='RGBA')
    for label, polygons in polygons_strctured.items():
        for polygon in polygons:
            draw_polygon(label, polygon, draw3)
    img3.save('seg_result3.png') #mask -> topo -> simplified polygon

    # draw original topo graph
    img4 = Image.open(img_path)
    draw4 = ImageDraw.Draw(img4, mode='RGBA')
    for label, polygons in polygons_strctured_0.items():
        for polygon in polygons:
            draw_polygon(label, polygon, draw4)
    img4.save('seg_result4.png') #mask -> topo polygon
