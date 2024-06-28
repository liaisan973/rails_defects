import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
start_time = time.time()
image = cv2.imread('datasetAnton/01_WheelBurn_Test_jpg.rf.44f39db113a48fecc4091571b17a198d.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start_point1 = (187, 0)
end_point1 = (145, 416)

start_point2 = (193, 0)
end_point2 = (288, 416)

mask = np.zeros_like(hsv)
cv2.fillPoly(mask, [np.array([start_point1, end_point1, end_point2, start_point2], dtype=np.int32)], 255)

defected_area = cv2.bitwise_and(hsv, mask)

lower_threshold = 0  # Lower threshold for dark pixels
upper_threshold = 150  # Upper threshold for dark pixels

dark_mask = cv2.inRange(defected_area, lower_threshold, upper_threshold)
detected = cv2.bitwise_and(image, image, mask=mask)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

cnts = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
area = 0
for c in cnts:
    area += cv2.contourArea(c)
    cv2.drawContours(image,[c], 0, (0,0,0), 2)
end_time = time.time()

processing_time = end_time - start_time

print("Processing Time: {:.4f} seconds".format(processing_time))

cv2.imwrite('mask.png', dark_mask)
cv2.imwrite('original.png', image)
cv2.imwrite('opening.png', opening)
cv2.imwrite('detected.png', detected)
cv2.imwrite('defected_area.png', defected_area)
cv2.waitKey(0)
cv2.destroyAllWindows()