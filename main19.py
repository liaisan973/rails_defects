import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import glob


def process_image(image_path, output_dir):
    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the points for the mask
    start_point1 = (197, 69)
    end_point1 = (174, 415)
    start_point2 = (199, 69)
    end_point2 = (220, 415)

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
        result = "This image has significant defects!"
    else:
        result = "This image has few defects."

    # Save the resulting images
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_mask.png'), dark_mask)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_original.png'), image)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_opening.png'), opening)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_detected.png'), detected)

    return (image_path, result)

def main(image_paths, output_dir):
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_path, output_dir) for image_path in image_paths]

        for future in as_completed(futures):
            image_path, result = future.result()
            print(f"Processed {image_path}: {result}")

    print(f"Processing completed in {time.time() - start_time} seconds")

if __name__ == "__main__":
    image_paths = [
        'datasetAnton/001_WheelBurn_Train_jpg.rf.86ea8b752d223b3a7ef8a23587dfe30b.jpg', 'datasetAnton/02_WheelBurn_Test_jpg.rf.4cc97a9a2ce1eb9dd1320b38087c0bb6.jpg', 'datasetAnton/003_WheelBurn_Train_jpg.rf.eac8617990986e571f8a892524e1b7a9.jpg', 'datasetAnton/004_WheelBurn_Train_jpg.rf.7e811355b121768ccb9b10bf4cbc0290.jpg'

    ]
    output_dir = 'output'
    main(image_paths, output_dir)


#images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]
