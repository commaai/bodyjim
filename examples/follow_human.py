#!/usr/bin/env python3
import cv2
import numpy as np
import ast
import os
import argparse
import onnx
from pathlib import Path
import pygame
import onnxruntime as ort

from bodyjim import BodyEnv

# following parameters
WIDTH_THRESHOLD = 0.20
CENTER_THRESHOLD = 0.1

# YOLOv5 implementation parameters
INPUT_SHAPE = (640, 416)
OUTPUT_SHAPE = (1, 16380, 85)
MODEL_PATH = Path(__file__).parent / 'yolov5n_flat.onnx'


def xywh2xyxy(x):
  y = x.copy()
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
  return y


def non_max_suppression(boxes, scores, threshold):
  # adapted from https://gist.github.com/CMCDragonkai/1be3402e261d3c239a307a3346360506
  assert boxes.shape[0] == scores.shape[0]
  ys1 = boxes[:, 0]
  xs1 = boxes[:, 1]
  ys2 = boxes[:, 2]
  xs2 = boxes[:, 3]
  areas = (ys2 - ys1) * (xs2 - xs1)
  scores_indexes = scores.argsort().tolist()
  boxes_keep_index = []
  while len(scores_indexes):
    index = scores_indexes.pop()
    boxes_keep_index.append(index)
    if not len(scores_indexes):
      break
    ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
    filtered_indexes = set((ious > threshold).nonzero()[0])
    scores_indexes = [
      v for (i, v) in enumerate(scores_indexes)
      if i not in filtered_indexes
    ]
  return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
  # adapted from https://gist.github.com/CMCDragonkai/1be3402e261d3c239a307a3346360506
  assert boxes.shape[0] == boxes_area.shape[0]
  ys1 = np.maximum(box[0], boxes[:, 0])
  xs1 = np.maximum(box[1], boxes[:, 1])
  ys2 = np.minimum(box[2], boxes[:, 2])
  xs2 = np.minimum(box[3], boxes[:, 3])
  intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
  unions = box_area + boxes_area - intersections
  ious = intersections / unions
  return ious


def nms(prediction, conf_thres=0.3, iou_thres=0.45):
  prediction = prediction[prediction[..., 4] > conf_thres]
  boxes = xywh2xyxy(prediction[:, :4])
  res = non_max_suppression(boxes, prediction[:, 4], iou_thres)
  result_boxes = []
  result_scores = []
  result_classes = []
  for r in res:
    result_boxes.append(boxes[r])
    result_scores.append(prediction[r, 4])
    result_classes.append(np.argmax(prediction[r, 5:]))
  return np.c_[result_boxes, result_scores, result_classes]


class YoloRunner:
  def __init__(self):
    onnx_path = MODEL_PATH
    if not onnx_path.exists():
      yolo_url = 'https://github.com/YassineYousfi/yolov5n.onnx/releases/download/yolov5n.onnx/yolov5n_flat.onnx'
      os.system(f'wget -P {onnx_path.parent} {yolo_url}')

    model = onnx.load(onnx_path)
    class_names_dict = ast.literal_eval(model.metadata_props[0].value)
    self.class_names = [class_names_dict[i] for i in range(len(class_names_dict))]

    self.output = np.zeros(np.prod(OUTPUT_SHAPE), dtype=np.float32)
    providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    self.session = ort.InferenceSession(onnx_path, providers=providers)

  def preprocess_image(self, img):
    img = cv2.resize(img, INPUT_SHAPE)
    img = np.expand_dims(img, 0).astype(np.float32) # add batch dim
    img = img.transpose(0, 3, 1, 2) # NHWC -> NCHW
    img /= 255 # 0-255 -> 0-1
    return img

  def run(self, img):
    img = self.preprocess_image(img)
    outputs = self.session.run(None, {'image': img})
    res = nms(outputs[0].reshape(1, 16380, 85))
    return [
      {
        "pred_class": self.class_names[int(opt[-1])],
        "prob": opt[-2],
        "pt1": opt[:2].astype(int).tolist(),
        "pt2": opt[2:4].astype(int).tolist()
      }
      for opt in res
    ]


def run_follow_human(body_ip):
  env = BodyEnv(body_ip, ["driver"], [], render_mode="human")
  obs, _ = env.reset()

  yolo = YoloRunner()

  while True:
    env.render()

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        env.close()
        return

    # run yolo on driover camera
    img = obs["cameras"]["driver"]
    outputs = yolo.run(img)

    # find the largest person in the frame
    max_area = None
    index_to_follow = None
    for i, output in enumerate(outputs):
      if output["pred_class"] != "person":
        continue
      if output["prob"] < 0.5:
        continue

      pt1, pt2 = output["pt1"], output["pt2"]
      w, h = abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1])
      area = w * h
      if max_area is None or area > max_area:
        max_area = area
        index_to_follow = i

    # move left to right if human is off-center
    # move towards the human if they are close to the center
    action = [0.0, 0.0]
    if index_to_follow is not None:
      person = outputs[index_to_follow]
      pt1, pt2 = person["pt1"], person["pt2"]     
      center_x, width = pt1[0] + (pt2[0] - pt1[0]) / 2, abs(pt1[0] - pt2[0])
      center_frac, width_frac = center_x / INPUT_SHAPE[0], width / INPUT_SHAPE[0]
      if width_frac < WIDTH_THRESHOLD:
        if abs(center_frac - 0.5) < CENTER_THRESHOLD:
          # go straight
          action = [-1.0, 0.0]
        else:
          if center_frac < 0.5:
            action = [0.0, 1.0]
          else:
            action = [0.0, -1.0]
    
    print("Executing action:", action)
    obs, _, _, _, _ = env.step(action)


if __name__=="__main__":
  parser = argparse.ArgumentParser("Human follower")
  parser.add_argument("body_ip", help="IP address of the body")
  args = parser.parse_args()

  run_follow_human(args.body_ip)
