import time
import cv2

start_time = time.time()

image = cv2.imread('rail3.JPG')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bilateral_blur = cv2.bilateralFilter(gray, 9, 75, 75)

_, thresh = cv2.threshold(bilateral_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

edges = cv2.Canny(bilateral_blur, 100, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

for contour in contours:
    if cv2.contourArea(contour) > 500:  # Filter out small contours
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

print(f"Number of potential defects detected: {len(contours)}")

end_time = time.time()

processing_time = end_time - start_time

print("Processing Time: {:.4f} seconds".format(processing_time))

# cv2.imwrite("index_contours_5.png", contours_image)
