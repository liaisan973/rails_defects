import cv2

# Read the image
image = cv2.imread('rail1.JPG')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("index_gray.png", gray)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imwrite("index_blur.png", blurred)

# Apply bilateral filter to preserve edges
bilateral_blur = cv2.bilateralFilter(gray, 9, 75, 75)
cv2.imwrite("index_bilateral.png", bilateral_blur)

# Threshold the image
_, thresh = cv2.threshold(bilateral_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("index_thresh.png", thresh)

# Detect edges using Canny
edges = cv2.Canny(bilateral_blur, 100, 200)
cv2.imwrite("index_edges.png", edges)

# Detect contours which might indicate defects
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contours_image = image.copy()
new_contours = cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
cv2.imwrite("index_contours.png", new_contours)

# Highlight defects on the original image
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Filter out small contours
        x, y, w, h = cv2.boundingRect(contour)
        defects = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite("index_defects.png", defects)



print(f"Number of potential defects detected: {len(contours)}")

