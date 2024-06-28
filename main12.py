import cv2
import numpy as np
import time
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import cv2
import supervision as sv
from ultralytics import YOLO

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cuda:0", # or 'cpu'
)

img = cv2.imread("demo_data/cars.jpg", cv2.IMREAD_UNCHANGED)
img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
numpydata = asarray(img_converted)
visualize_object_predictions(
    numpydata,
    object_prediction_list = result.object_prediction_list,
    hide_labels = 1,
    output_dir='/content/demo_data',
    file_name = 'result',
    export_format = 'png'
)
Image('demo_data/result.png')

# image = cv2.imread('DSC04371.JPG')
# model = YOLO("yolov8n.pt")
#
# def callback(image_slice: np.ndarray) -> sv.Detections:
#     result = model(image_slice)[0]
#     return sv.Detections.from_ultralytics(result)
#
# slicer = sv.InferenceSlicer(callback = callback)
#
# detections = slicer(image)
