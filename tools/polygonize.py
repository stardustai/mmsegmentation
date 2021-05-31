from math import radians
from numpy.lib.arraysetops import isin
from numpy.lib.polynomial import poly
import topojson as tp
import pickle, random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
# from imantics import Polygons, Mask, Annotation
from geojson import Feature, Polygon, FeatureCollection, LineString, MultiLineString, MultiPolygon
from skimage import measure
from skimage.feature import canny
import skimage.morphology as sm
from collections import defaultdict
from tqdm import tqdm
from math import sin, cos, atan, pi
from imantics import Mask
import topojson as tp
from shapely import geometry
import topojson as tp
# import geopandas as gpd

# world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
# data = world.query('continent == "Africa"')
# r = tp.Topology(data, topology=True).toposimplify(4).to_alt().properties(title='WITH Topology')

palette = eval(open('data/parkinglot/color.json', 'r').read())
CLASSES = ('road', 'curb', 'obstacle', 'chock', 'parking_line', 'road_line', 'vehicle')
PALETTE = [(0, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 128, 255), (128, 128, 128)]
tolerance = 1
draw_img = True
img_path = 'temp/test.jpg'
save_svg = False


def polygonize(result, tolerance=1, draw_img=None):
    # convert bitmap to polygon
    polygons_data = extractPolygons(result)
    # sort by area from large to small
    polygons_data.sort(key=lambda p: p['area'], reverse=True)
    # join vertex of polygons
    snapPolygonPoints(polygons_data, result)  # snap points

    #simplify polygons using topo
    topo_data = [Feature(
        geometry=Polygon([p['polygon']]),
        properties={"name": p['label']}
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


def applyMorph(result):
    ## Image morphic operation
    # The morphological opening on an image is defined as an erosion followed by a dilation. 
    # Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks. 
    # This tends to “open” up (dark) gaps between (bright) features.
    result = sm.opening(result, sm.disk(2))  #用边长为2的圆形滤波器进行膨胀滤波
    result = sm.dilation(result,sm.disk(tolerance))  #用边长为5的正方形滤波器进行膨胀滤波
    result = sm.closing(result, sm.disk(2))
    return result


#util
def getBoxAreaAndCenter(points):
    center = (points[:,0].mean(), points[:,1].mean())
    area = cv2.contourArea(points.astype(np.float32))
    return area, center


def extractPolygons(result):
    # find polygon
    temp_polygons = []
    polygons_data = []
    for label in palette:
        i = CLASSES.index(label)
        mask = np.where(result==i, 1, 0).astype(np.uint8)
        # using imantics
        # polygons = Mask(mask).polygons().points 

        # Using CV2
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        polygons = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
        polygons = polygons[0] if len(polygons) == 2 else polygons[1]
        polygons = [polygon.squeeze() for polygon in polygons]

        # using skimage
        # mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        # polygons = measure.find_contours(mask, 0.8) 
        # polygons = [np.vstack([p[:,1]-1, p[:,0]-1]).T for p in polygons]#reverse for skimage

        print(f'{label}:{len(polygons)}')
        for j, polygon in enumerate(polygons):
            # remove duplicated polygon
            if not any([polygon.shape == p.shape and polygon.sum() == p.sum() for p in temp_polygons]):
                temp_polygons.append(polygon)
            else:
                print(f'find duplicated polygon: {label}({j})')
                continue
            if polygon.shape[0] <= 2:
                continue
            area, center = getBoxAreaAndCenter(polygon)
            polygon_approximated = measure.approximate_polygon(polygon, tolerance)
            label2 = label+str(j)
            if area >= 100:
                polygons_data.append({
                    'label': label,
                    'polygon': polygon.tolist(),
                    # 'topo': polygon_feat,
                    'approximated': polygon_approximated,
                    'area': area,
                    'center': center,
                    'label2': label2
                })
            else:
                print(f'---> Small region {label} with area:{area} and length:{polygon.shape[0]}')
    return polygons_data
    
def snapPolygonPoints(polygons_data:list, mask:np.ndarray):
    ''' create a lookup table to register polygon vertices
    with value on [x, y, 1]-> i'th polygon and 
    with value on [x, y, 2]-> j'th coordinates of i'th polygon
    '''

    vertices_lookup = np.full(list(mask.shape[::-1])+[2], fill_value=-1) 
    polygons = [i['polygon'] for i in polygons_data]
    labels = [i['label'] for i in polygons_data]
    label_indices = [CLASSES.index(l) for l in labels]

    #check polygon joint
    checkJointFromPolygon(polygons, mask)

    # utils
    radius = 1
    x_upper, y_upper = mask.shape[::-1]
    _validate_coord = lambda x,y: x >= 0 and y >= 0 and x < x_upper and y < y_upper
    _lookup = lambda x,y,i=0: vertices_lookup[x, y, i] if _validate_coord(x,y) else None
    _lookup_check_pair = lambda x,y,label: _lookup(x, y) is not None and _lookup(x, y) != -1 and _lookup(x, y) != label
    # _lookup_range = lambda x0, x1, y0, y1: np.array([[_lookup(x, y) for x in range(x0,x1)] for y in range(y0, y1)])
    _lookup_scope = lambda x, y: np.array([[_lookup(x, y) for x in range(x-3, x+4)] for y in range(y-3, y+4)])
    _lookup_mask = lambda x,y: mask[y,x] if _validate_coord(x,y) else None # need to reverse x,y from OpenCV to NumPy
    _mask_scope = lambda x,y: np.array([[_lookup_mask(x_,y_) for x_ in range(x-3, x+4)] for y_ in range(y-3, y+4)])
    _point_distance = lambda x1, y1, x2, y2: abs(x2-x1) + abs(y2-y1)
    def _get_valid_points(x, y, label_index, radius = 1):
        valid_points = set()
        # search valid points from offsets
        offsetss = [[(i,j) for i in range(-radius, radius+1)] for j in range(-radius, radius+1)]
        for offsets in offsetss:
            for offset in offsets:
                x_i, y_i = x+offset[0], y+offset[1]
                if _point_distance(x,y, x_i, y_i) > radius or (x,y)==(x_i,y_i):
                    continue
                if _lookup_check_pair(x_i, y_i, label_index):
                    valid_points.add((x_i, y_i))
        return valid_points

    # create lookup table
    for i, polygon in enumerate(polygons):
        for j, (x, y) in enumerate(polygon):
            x, y = round(x), round(y)
            assert _validate_coord(x, y)
            if _lookup(x, y) != -1:
                vertices_lookup[x, y] = -1
                continue
            vertices_lookup[x, y, :] = [i,j]

    points_snaped = []
    points_missed = []
    # search for vertices to merge
    for i, polygon in tqdm(enumerate(polygons)):
        label_index = label_indices[i]
        for j, (x, y) in enumerate(polygon):
            x, y = round(x), round(y)
            if _lookup(x, y) == -1:
                continue #snapped
            assert _lookup(x, y) == i
            x2, y2 = polygon[(j+1)%len(polygon)]#next point
            x2, y2 = round(x2), round(y2)
            # get target coordinate
            # check whether mask is outside and
            # check whether candidate coordinates next another polygon's vertex
            x_t, y_t = None, None
            valid_points = _get_valid_points(x, y, i)
            # check for valid points
            if len(valid_points) >=2:
                valid_points2 = _get_valid_points(x2, y2, i)
                vp1 = valid_points - valid_points2
                # vp2 = valid_points2 - valid_points
                vp3 = valid_points & valid_points2
                if len(vp1)>=1 and len(valid_points2)>=1:
                    valid_points = vp1
                elif len(vp3) == 2:
                    valid_points = {vp3.pop()}
                else:
                    valid_points = set()
            # exception
            if len(valid_points) == 0:
                # print(f'No adjacent vertice found at [{i}]({x},{y}):\n{_lookup_scope(x, y)}')
                points_missed.append((i, j, (x, y)))
                continue
            #snap points
            valid_points.add((x,y))
            vertices = np.array(list(valid_points))
            x_a, y_a = vertices.mean(axis = 0).tolist()
            for x_t, y_t in valid_points:
                j_t = _lookup(x_t, y_t) #find j'th polygon and n'th coordinates
                n_t = _lookup(x_t, y_t, 1) #find n'th point
                # if (x,y) != (x_t, y_t):
                #     print(f'Found adjacent vertice [{i}]({x},{y})<->[{j_t}]({x_t},{y_t})')
                polygons[j_t][n_t] = [x_a, y_a]
                # remove from lookup table
                vertices_lookup[x_t, y_t, 0] = -1
                points_snaped.append((j_t, n_t, (x, y)))
    #make sure all points are cleared
    print(f'{len(points_snaped)} points snapped and {len(points_missed)} points left ({len(points_snaped)/(len(points_missed)+len(points_snaped))*100:.1f}%)')


def get_polygon_dict(topo):
    polygons_strctured = defaultdict(list)
    topo_json = eval(topo.to_geojson())
    for geo in topo_json['features']:
        label = geo['properties']['name']
        polygon = geo['geometry']['coordinates'][0]
        polygons_strctured[label].append(polygon)
    return polygons_strctured


def checkJointFromPolygon(polygons, mask, draw_single_points=False):
    point_count = np.zeros(mask.shape[::-1])
    for i, polygon in enumerate(polygons):
        polygon_np = np.zeros_like(point_count)
        for (x,y) in polygon:
            _x, _y = round(x), round(y)
            point_count[_x, _y] += 1
            polygon_np[_x, _y] = 1
        # Image.fromarray(np.where(polygon_np==1, 255, 0).astype(np.uint8)).save(f'tmp/{i}.png')
    singles = (point_count == 1).sum()
    point_o = point_count.sum() - singles
    # single_p_np = np.zeros_like(point_count)
    # single_p_np[point_count == 1] = 255
    if draw_single_points:
        draw_np = np.where(point_count>1, 128, 0)
        draw_np[point_count == 1] = 255
        Image.fromarray(draw_np.astype(np.uint8), mode='P').save('temp/single_points.png')
    print(f'There are {point_o} points joined({point_o/(singles+point_o)*100:.1f}%)')


# draw image for debugging
def draw_polygon(label, color, polygon, draw):
    # fnt = ImageFont.truetype("Arial.ttf", 20)
    p = [tuple(i) for i in polygon]
    polygon = np.array(polygon)
    center = [polygon[:,0].mean(), polygon[:,1].mean()]
    center[0] -= 5*len(label)
    center[1] -= 10
    color2 = tuple(list(color)+[128]) #transparent
    color3 = tuple([150-c for c in color]) #invert
    # draw.point(polygon3, fill=color3)
    # draw.line(polygon3, width=1, fill=color, joint='curve')
    draw.polygon(p, fill=color2, outline=color)
    draw.point(p, fill=color3)
    draw.text(center, label, fill=color)

def drawResults(polygons_data, img_path, mode=None):
    img = Image.open(img_path)
    # im = img.convert(mode='RGBA')
    draw = ImageDraw.Draw(img, mode='RGBA')
    # im2 = img.convert(mode='RGBA')
    img2 = img.copy()
    draw2 = ImageDraw.Draw(img2, mode='RGBA')
    # draw approximated polygon (inferior)
    img_path = img_path + f'_result_mode{mode}.png'
    if mode == 1: # draw original prediction
        for data in polygons_data:
            label = data['label']
            label2 = data['label2']
            polygon1 = data['polygon']
            color = PALETTE[CLASSES.index(label)]
            draw_polygon(label2, color, polygon1, draw)
        img.save(img_path) #mask -> polygon

    elif mode == 2: # draw prediction using approximation
        for data in polygons_data:
            label = data['label']
            label2 = data['label2']
            polygon2 = data['approximated']
            color = PALETTE[CLASSES.index(label)]
            draw_polygon(label2, color, polygon2, draw2)
        img2.save() #mask -> approximate polygon

    # draw image: mask -> topo -> simplified polygon
    elif mode == 3:
        img3 = img.copy()
        draw3 = ImageDraw.Draw(img3, mode='RGBA')
        for label, polygons in polygons_data.items():
            for polygon in polygons:
                color = PALETTE[CLASSES.index(label)]
                draw_polygon(label, color, polygon, draw3)
        img3.save(img_path) #mask -> topo -> simplified polygon

    # draw original topo graph
    elif mode == 4:
        # img4 = Image.fromarray(np.zeros_like(result.astype(np.uint8))).convert(mode='RGB')
        img4 = img.copy()
        draw4 = ImageDraw.Draw(img4, mode='RGBA')
        for label, polygons in polygons_data.items():
            for polygon in polygons:
                color = PALETTE[CLASSES.index(label)]
                draw_polygon(label, color, polygon, draw4)
        img4.save(img_path) #mask -> topo polygon
    
    else:
        raise Exception(f'Unexpected mode: {mode}')

    return img_path



if __name__ == '__main__':
    ## Load data
    # load from topo_data
    # data = pickle.load(open('topo_data', 'rb')) 
    # load from mask
    polygons_data = []
    img = Image.open('temp/mask.png')
    result = np.asarray(img)
    polygon_data = extractPolygons(result)

    #sort by area from large to small
    polygons_data.sort(key=lambda p:p['area'], reverse=True)
    snapPolygonPoints(polygons_data, result)#snap points

    #simplify polygons using topo
    topo_data = [Feature(
                geometry = Polygon([p['polygon']]),
                properties = {"name": p['label']}
                ) for p in polygons_data]
    # pickle.dump(topo_data, open('topo_data', 'wb'))
    fc = FeatureCollection(topo_data)
    topo = tp.Topology(fc, prequantize=True, topology=True, shared_coords=True)
    topo_s = topo.toposimplify(
        epsilon=tolerance, 
        simplify_algorithm='dp', 
        # simplify_algorithm='vw', 
        # simplify_with='simplification', 
        )

    if save_svg:
        with open('temp/topo.svg', 'w') as f:
            f.write(topo_s.to_svg())
        with open('temp/topo0.svg', 'w') as f:
            f.write(topo.to_svg())


    polygons_strctured = get_polygon_dict(topo_s)
    polygons_strctured_0 = get_polygon_dict(topo)

    polygons = [p for l,ps in polygons_strctured.items() for p in ps]
    checkJointFromPolygon(polygons, result, True)

    if draw_img:
        drawResults(polygons_data, img_path)
        
