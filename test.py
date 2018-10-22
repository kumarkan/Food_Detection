# Author : Anil Kumar (anil912001@gmail.com)
# Modified From API
# https://github.com/wagonhelm/TF_ObjectDetection_API/blob/master/ChessObjectDetection.ipynb

import skimage
import numpy as np
from skimage import io, transform
import os
import shutil
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import urllib.request
import urllib.error
import cv2

from utils.app_utils import FPS, WebcamVideoStream
from utils import label_map_util
from utils import visualization_utils as vis_util

root = os.getcwd()
imagePath = os.path.join(root, 'images')
labelsPath = os.path.join(root, 'labels')
linksPath = os.path.join(imagePath, 'imageLinks')
trainPath = os.path.join(imagePath, 'train')
testPath = os.path.join(imagePath, 'test')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'object_detection_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/label_map.pbtxt'

NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Modified From API
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    #src=0 --> /dev/video0. change src number according to your video node
    video_capture = WebcamVideoStream(src=0, 
                                      width=720,
                                      height=480).start()
    fps = FPS().start()
    while(True):
      frame =  video_capture.read()
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image_np_expanded = np.expand_dims(frame_rgb, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      print('***************** Visualization ************')
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          frame_rgb,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=2)
      
      output_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
      cv2.imshow('Object Detection', output_rgb)
      fps.update()
      if cv2.waitKey(1) & 0xFF == ord('q'):
      	break


