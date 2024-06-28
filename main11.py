import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()

# Read the input image
image = cv2.imread('DSC04371.JPG')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start_point1 = (0, 199)
end_point1 = (1280, 217)

start_point2 = (0, 720)
end_point2 = (1280, 744)

mask = np.zeros_like(gray)
cv2.fillPoly(mask, [np.array([start_point1, end_point1, end_point2, start_point2], dtype=np.int32)], 255)

defected_area = cv2.bitwise_and(gray, mask)

orb = cv2.ORB_create()

kp = orb.detect(defected_area,None)

kp, des = orb.compute(defected_area, kp)

img2 = cv2.drawKeypoints(defected_area, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
print(kp)




# cv2.imwrite('detected_lines.jpg', defected_area)
end_time = time.time()
#
# # Calculate processing time per frame
processing_time = end_time - start_time
#
# # Calculate FPS
# #fps = 1 / processing_time
#
print("Processing Time: {:.4f} seconds".format(processing_time))
# #print("FPS: {:.2f}".format(fps))
