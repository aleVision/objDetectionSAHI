from sahi.models import yolov8
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.utils.cv import read_image_as_pil
from sahi.utils.cv import visualize_object_predictions
from sahi.utils.file import download_from_url
from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)

import cv2
import numpy as np
from PIL import Image

yolov5_model_path = 'models/yolov5s6.pt'
#download_yolov5s6_model(destination_path=yolov5_model_path)
#download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')

yolov8_model_path = 'yolov8s.pt'
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=yolov8_model_path, confidence_threshold=0.4, device="cuda:0")

# result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)
result = get_sliced_prediction('demo_data/small-vehicles1.jpeg', detection_model, slice_height = 256, slice_width = 256, 
                               overlap_height_ratio = 0.2, overlap_width_ratio = 0.2)

img = cv2.imread('demo_data/small-vehicles1.jpeg', cv2.IMREAD_UNCHANGED)
img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
numpydata = np.asarray(img_converted)

visualize_object_predictions(numpydata, object_prediction_list = result.object_prediction_list,
                             hide_labels = True, file_name = 'result', export_format = 'png', text_size=0.01, text_th=0.01)

result.export_visuals(export_dir="result/sahi.png", hide_labels=True, hide_conf=True)
#Image('result/prediction_visual.png')