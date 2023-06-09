
# Necessary Imports
import os, requests, torch, math, cv2, sys, PIL, argparse, dlib,imutils, time, json
import numpy as np
from datetime import datetime
from typing import List, Optional
from imutils.video import VideoStream
from imutils.video import FPS

# Imports for YoloV6 for Object Detection
from YOLOv6 import yolov6
sys.modules['yolov6'] = yolov6

from YOLOv6.yolov6.utils.events import LOGGER, load_yaml
from YOLOv6.yolov6.layers.common import DetectBackend
from YOLOv6.yolov6.data.data_augment import letterbox
from YOLOv6.yolov6.utils.nms import non_max_suppression
from YOLOv6.yolov6.core.inferer import Inferer




print('Imported all libraries and frameworks.')

#Set-up hardware options
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
         
def check_img_size(img_size, s=32, floor=0):
  def make_divisible( x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor
  """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
  if isinstance(img_size, int):  # integer i.e. img_size=640
      new_size = max(make_divisible(img_size, int(s)), floor)
  elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
      new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
  else:
      raise Exception(f"Unsupported type of img_size: {type(img_size)}")

  if new_size != img_size:
      print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
  return new_size if isinstance(img_size,list) else [new_size]*2

def precess_image(path, img_size, stride):
  '''Process image before image inference.'''
  try:
    from PIL import Image
    if type(path) == str:
      img_src = np.asarray(Image.open(path))
    else:
      img_src = path
    assert img_src is not None, f'Invalid image: {path}'
  except Exception as e:
    LOGGER.Warning(e)
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src

model_sm = DetectBackend(f"models/yolov6n.pt", device=device)
stride = model_sm.stride
class_names = load_yaml("YOLOv6/data/coco.yaml")['names']

model_sm.model.float()

img_size = (640,640)
if device.type != 'cpu':
  model_sm(torch.zeros(1, 3, *img_size).to(device).type_as(next(model_sm.model.parameters())))  # warmup



model_lrg = DetectBackend(f"models/yolov6l.pt", device=device)
stride = model_lrg.stride
class_names = load_yaml("YOLOv6/data/coco.yaml")['names']

model_lrg.model.float()

img_size = (640,640)
if device.type != 'cpu':
  model_lrg(torch.zeros(1, 3, *img_size).to(device).type_as(next(model_lrg.model.parameters())))  # warmup




def detect(image, model, conf_thresh, iou_thresh):

  hide_labels: bool = False 
  hide_conf: bool = False 

  img_size:int = 640

  conf_thres: float = conf_thresh
  iou_thres: float = iou_thresh
  max_det:int =  1000
  agnostic_nms: bool = False 

  img_size = check_img_size(img_size, s=stride)

  img, img_src = precess_image(image, img_size, stride)
  img = img.to(device)
  if len(img.shape) == 3:
      img = img[None]
      # expand for batch dim
  pred_results = model(img)
  classes:Optional[List[int]] = None # the classes to keep
  det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

  gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
  img_ori = img_src.copy()
  if len(det):
    det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
  return det




def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 255, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    font_scale = 1
    font = cv2.FONT_HERSHEY_PLAIN
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), text_color_bg, -1)
    cv2.putText(img, text, (x+5, y-5), font, font_scale, text_color, font_thickness)


def draw_bb_text(frame, text,
          bbox,
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 255, 0),
          font_thickness=2,
          text_color_bg=(255, 255, 255)
          ):

    startX, startY, endX, endY = bbox
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    startY = 20 if startY < 20 else startY
    startX = 1 if startX < 1 else startX
    bg = np.ones_like(frame[startY-20:startY,startX-1:startX+text_w+3]).astype('uint8') * 255
    bg[:,:] = text_color_bg
    frame[startY-20:startY,startX-1:startX+text_w+3] = cv2.addWeighted(frame[startY-20:startY,startX-1:startX+text_w+3], 0.0, bg, 1.0, 1)
    cv2.putText(frame, text, (startX, startY-text_h+2), font, font_scale, text_color, font_thickness)



def get_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union = box1_area + box2_area - intersection

    return intersection / union



def get_color(idx):
    idx = idx * 3
    color = (int((37 * idx) % 255), int((17 * idx) % 255), int((29 * idx) % 255))

    return color