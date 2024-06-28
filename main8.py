import cv2
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread('DSC03044.JPG')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bilateral_blur = cv2.bilateralFilter(gray, 9, 75, 75)

_, thresh = cv2.threshold(bilateral_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

edges = cv2.Canny(bilateral_blur, 100, 200)

h, theta, d = hough_line(edges)

# Extract lines using probabilistic Hough Transform
lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

# Display the Hough transform result
plt.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]], cmap='gray', aspect=1/30)
plt.title('Hough Transform')
plt.xlabel('Angles (degrees)')
plt.ylabel('Distance (pixels)')
plt.show()

# Display the detected lines
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
plt.title('Detected Lines')
plt.show()