import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import glob

from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)
import yolov5

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

yolov5_model_path = 'yolov5/runs/train/exp/weights/best.pt'
download_yolov5s6_model(destination_path=yolov5_model_path)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5_custom',
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
)


def process_image(image_path):
    start_time = time.time()
    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the points for the mask
    start_point1 = (0, 186)
    end_point1 = (415, 128)
    start_point2 = (0, 218)
    end_point2 = (415, 196)

    # Create the mask
    mask = np.zeros_like(hsv)
    cv2.fillPoly(mask, [np.array([start_point1, end_point1, end_point2, start_point2], dtype=np.int32)], 255)

    # Apply the mask to the image
    defected_area = cv2.bitwise_and(hsv, mask)

    # Thresholding to find dark areas
    lower_threshold = 0  # Lower threshold for dark pixels
    upper_threshold = 100  # Upper threshold for dark pixels
    dark_mask = cv2.inRange(defected_area, lower_threshold, upper_threshold)
    detected = cv2.bitwise_and(image, image, mask=mask)

    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Initialize metrics
    total_defect_area = 0
    num_defects = len(cnts)
    max_defect_area = 0

    for c in cnts:
        area = cv2.contourArea(c)
        total_defect_area += area
        if area > max_defect_area:
            max_defect_area = area
        cv2.drawContours(image, [c], 0, (0, 0, 0), 2)

    # Average defect size
    average_defect_area = total_defect_area / num_defects if num_defects > 0 else 0

    # Define thresholds
    threshold_total_defect_area = 1000  # Set your desired threshold here
    threshold_num_defects = 10  # Set your desired threshold here
    threshold_max_defect_area = 500  # Set your desired threshold here
    threshold_average_defect_area = 100  # Set your desired threshold here

    # Evaluate the image based on the metrics
    if (total_defect_area > threshold_total_defect_area or
        num_defects > threshold_num_defects or
        max_defect_area > threshold_max_defect_area or
        average_defect_area > threshold_average_defect_area):
        print("This image has significant defects!")

        result_sahi = get_sliced_prediction(image_path, detection_model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        result_sahi.export_visuals(export_dir="demo_data/")

        Image("demo_data/prediction_visual1.png")


    else:
        print("This image has few defects.")

    print(f"Processing completed in {time.time() - start_time} seconds")


    # Save the resulting images
    # base_filename = os.path.splitext(os.path.basename(image_path))[0]
    # cv2.imwrite(os.path.join(output_dir, f'{base_filename}_mask.png'), dark_mask)
    # cv2.imwrite(os.path.join(output_dir, f'{base_filename}_original.png'), image)
    # cv2.imwrite(os.path.join(output_dir, f'{base_filename}_opening.png'), opening)
    # cv2.imwrite(os.path.join(output_dir, f'{base_filename}_detected.png'), detected)

    # return (image_path, result_sahi)

# def main(image_paths, output_dir):
#     start_time = time.time()
#
#     with ProcessPoolExecutor() as executor:
#         futures = [executor.submit(process_image, image_path, output_dir) for image_path in image_paths]
#
#         for future in as_completed(futures):
#             image_path, result = future.result()
#             print(f"Processed {image_path}: {result}")
#
#     print(f"Processing completed in {time.time() - start_time} seconds")


if __name__ == "__main__":

    # output_dir = 'output'
    process_image("yolov5/rails_defects-4/test/images/050_Spalling_Train_png.rf.cc7b8a1b2cb9e552f46e27944e20bc36.jpg")


#images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]
