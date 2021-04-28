import flask
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import object_detection
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
#sys.path.append("..")


detection_model = tf.saved_model.load('saved_model')
# from utils import visualization_utils as vis_util
MODEL_NAME = 'saved_model'
PATH_TO_CKPT = MODEL_NAME + '/saved_model.pb'
PATH_TO_LABELS = os.path.join('Letters_label_map.pbtxt')
NUM_CLASSES = 26


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_fn(image):
    model = detection_model.signatures['serving_default']
    detections = model(image)
    return detections

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)
    for img in TEST_IMAGE_PATHS:
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
        im = Image.fromarray(image_np_with_detections)
        im.save('uploads/' + filename)

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
