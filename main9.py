import cv2
import numpy as np

# Read the input image
image = cv2.imread('DSC04371.JPG')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to find edges
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use the Hough line transform to detect lines in the edge-detected image
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

# Draw the detected horizontal line on the original image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        if np.pi/2 - 0.1 < theta < np.pi/2 + 0.1:  # Check for horizontal lines with some tolerance
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display or save the image with the detected horizontal line
cv2.imwrite('detected_lines.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
