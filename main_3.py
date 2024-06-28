import cv2
import time

# Load the image
image = cv2.imread('DSC04371.JPG')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("index_gray.png", gray)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

cv2.imwrite("index_blur.png", blurred)

bilateral_blur = cv2.bilateralFilter(gray, 9, 75, 75)

cv2.imwrite("index_bilateral.png", bilateral_blur)

thresh = cv2.threshold(bilateral_blur, 0, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite("index_thresh.png", thresh)

edges = cv2.Canny(bilateral_blur, 100, 200)

cv2.imwrite("index_edges.png", edges)

ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite("index_thresh2.png", th)


#
# # Edge detection using Canny
# edges = cv2.Canny(blurred, 50, 150)
#
# # Find contours
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Find the largest contour (assuming it corresponds to the metal stick)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Start time
# start_time = time.time()
#
# # Calculate the bounding box around the largest contour
# x, y, w, h = cv2.boundingRect(largest_contour)
#
# # Define the amount to increase the bounding box size (for width and height independently)
# increase_width = 10000
# increase_height = 480
#
# # Adjust the coordinates to increase the bounding box size
# x -= increase_width // 2
# y -= increase_height // 2
# w += increase_width
# h += increase_height
#
# # Ensure the bounding box stays within the image boundaries
# x = max(x, 0)
# y = max(y, 0)
# w = min(w, image.shape[1] - x)
# h = min(h, image.shape[0] - y)
#
# # Draw the rectangular bounding box on the original image
# boxed_image = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# # End time
# end_time = time.time()
#
# # Calculate processing time per frame
# processing_time = end_time - start_time
#
# # Calculate FPS
# #fps = 1 / processing_time
#
# print("Processing Time: {:.4f} seconds".format(processing_time))
# #print("FPS: {:.2f}".format(fps))
#
# # Display the image with the rectangular bounding box
# cv2.imshow('Metal Stick with Rectangular Bounding Box', boxed_image)
#
# # Optionally save the image with the rectangular bounding box
# cv2.imwrite('metal_stick_with_rectangular_bounding_box_4.jpg', boxed_image)
#
# # Wait for a key press and close all windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()