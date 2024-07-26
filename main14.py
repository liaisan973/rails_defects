import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# start_time = time.time()

# Read the input image
image = cv2.imread('IMG_20201114_100215.jpg')

# Convert the image to grayscale
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start_point1 = (0, 872)
end_point1 = (4000, 866)

start_point2 = (0, 1854)
end_point2 = (4000, 1873)

mask = np.zeros_like(hsv)
cv2.fillPoly(mask, [np.array([start_point1, end_point1, end_point2, start_point2], dtype=np.int32)], 255)

defected_area = cv2.bitwise_and(hsv, mask)

lower_threshold = 0  # Lower threshold for dark pixels
upper_threshold = 50  # Upper threshold for dark pixels
dark_mask = cv2.inRange(defected_area, lower_threshold, upper_threshold)
detected = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite('detected_lines5.jpg', defected_area)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)


cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
total_defect_area = 0
for c in cnts:
    total_defect_area += cv2.contourArea(c)
    cv2.drawContours(defected_area,[c], 0, (0,0,0), 2)
    cv2.drawContours(image, [c], 0, (60, 70, 100), 2)

threshold_defect_area = 100000  # Set your desired threshold here

if total_defect_area > threshold_defect_area:
    print("This image has a large number of defects!")
else:
    print("This image has few defects.")


cv2.imwrite('image.png', image)
cv2.imwrite('original.png', defected_area)
cv2.imwrite('opening.png', opening)
cv2.imwrite('detected.png', detected)
cv2.imwrite('mask.png', mask)
print(total_defect_area)

