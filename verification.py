from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

# config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
config_file = 'configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py'
# checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
checkpoint_file = 'checkpoints/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test_img/rgb_0.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# visualize the results in a new window
model.show_result(img, result, show=True)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#    result = inference_segmentor(model, frame)
#    model.show_result(frame, result, wait_time=1)