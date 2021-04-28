import os
import sys
sys.path.append("models/research")
import cv2 
import numpy as np
import tensorflow as tf
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder





category_index = label_map_util.create_category_index_from_labelmap('Letters_label_map.pbtxt')

detection_model = tf.saved_model.load('saved_model')
print(detection_model.signatures['serving_default'].inputs)

@tf.function
def detect_fn(image):
    model = detection_model.signatures['serving_default']
    detections = model(image)
    return detections

def predict(img):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  image_np = np.array(img)
  label_id_offset = 0
  input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0))
  detections = detect_fn(input_tensor)

  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
  detections['num_detections'] = num_detections
  # print(detections['num_detections'])

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  # print(detections['detection_classes']+label_id_offset)
  # print(detections['detection_scores'])
  # print(detections['detection_boxes'])


  image_np_with_detections = image_np.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    (detections['detection_classes']+label_id_offset).astype(int),
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=1,
    min_score_thresh=.2,
    line_thickness=8,
    agnostic_mode=False
    )
  # classes = detections['detection_classes']+label_id_offset
  # clas = str(classes[0])
  # print(alphabet[clas])
  # mytext = alphabet[clas]
  # print(mytext)

  image_np_with_detections= cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)

  return cv2.imencode('.png',image_np_with_detections)

