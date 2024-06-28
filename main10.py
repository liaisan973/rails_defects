import cv2
import numpy as np


# Read the input image
image = cv2.imread('DSC04371.JPG')

# Define the coordinates of the start and end points of the first line
start_point1 = (0, 199)  # Example coordinates, replace with your actual values
end_point1 = (1280, 217)    # Example coordinates, replace with your actual values

# Define the coordinates of the start and end points of the second line
start_point2 = (0, 720)  # Example coordinates, replace with your actual values
end_point2 = (1280, 744)    # Example coordinates, replace with your actual values

# Draw the first line on the image
color1 = (0, 0, 255)  # BGR color format, here we are using red (B=0, G=0, R=255)
thickness1 = 2        # Line thickness
image_with_lines = cv2.line(image, start_point1, end_point1, color1, thickness1)

# Draw the second line on the image
color2 = (255, 0, 0)  # BGR color format, here we are using blue (B=255, G=0, R=0)
thickness2 = 2        # Line thickness
image_with_lines = cv2.line(image_with_lines, start_point2, end_point2, color2, thickness2)

# Display or save the image with the drawn lines
cv2.imwrite('detected_lines4.jpg', image_with_lines)

cv2.waitKey(0)
cv2.destroyAllWindows()
