import cv2

# Load the image
image = cv2.imread('rail3.jpg')  # Replace 'metal_stick_photo.jpg' with your image filename

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it corresponds to the metal stick)
largest_contour = max(contours, key=cv2.contourArea)

# Calculate the bounding box around the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Draw the bounding box on the original image
boxed_image = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the bounding box
cv2.imshow('Metal Stick with Bounding Box', boxed_image)

# Optionally save the image with the bounding box
cv2.imwrite('metal_stick_with_bounding_box.jpg', boxed_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()