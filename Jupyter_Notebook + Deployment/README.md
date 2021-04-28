# FlaskObjectDetection - TensorFlow

## Project
- This project is worked with tools tensorflow version 1,
- When you install tensorflow v2, the project is already updated to directly call the tools of version 1
## Run Project
##### Install requirements
```
pip install -r requirements.txt
```
##### Download Model [frozen_inference_graph.pb](https://github.com/datitran/object_detector_app/blob/master/object_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb)
* Create folder ssd_mobilenet_v1_coco_11_06_2017
```
mkdir ssd_mobilenet_v1_coco_11_06_2017
```
* Copy frozen_inference_graph.pb and paste in folder ssd_mobilenet_v1_coco_11_06_2017
##### Run flask
```
flask run
```

