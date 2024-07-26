import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

image = cv2.imread('IMG_20201114_100215.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.bilateralFilter(hsv,9,75,75)

start_point1 = (0, 872)
end_point1 = (4000, 866)

start_point2 = (0, 1854)
end_point2 = (4000, 1873)

mask = np.zeros_like(blur)
cv2.fillPoly(mask, [np.array([start_point1, end_point1, end_point2, start_point2], dtype=np.int32)], 255)

defected_area = cv2.bitwise_and(blur, mask)

lower_threshold = 0  # Lower threshold for dark pixels
upper_threshold = 50  # Upper threshold for dark pixels
dark_mask = cv2.inRange(defected_area, lower_threshold, upper_threshold)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, cnts, 0, (0,255,0), 3)

# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# total_defect_area = 0
# for c in cnts:
#     total_defect_area += cv2.contourArea(c)
#     cv2.drawContours(image,[c], 0, (160,30,70), 2)

# print(total_defect_area)

cv2.imwrite('mask_22.png', image)







