import cv2
import numpy as np


def find_largest_horizontal_lines(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Image not found or unable to load."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    if lines is None:
        return None, "No lines detected"

    horizontal_lines = []

    # Filter out horizontal lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1)) * 180 / np.pi
            if angle < 10 or angle > 170:  # Consider lines with small angles
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                horizontal_lines.append((x1, y1, x2, y2, length))

    # Merge line segments that are close to each other and almost collinear
    def merge_lines(lines):
        if len(lines) == 0:
            return []

        # Sort lines by x1 coordinate
        lines = sorted(lines, key=lambda line: line[0])

        merged_lines = []
        current_line = lines[0]

        for next_line in lines[1:]:
            x1, y1, x2, y2, length = current_line
            nx1, ny1, nx2, ny2, nlength = next_line

            # Check if the lines are close to each other and almost collinear
            if np.abs(y1 - ny1) < 20 and np.abs(y2 - ny2) < 20:
                current_line = (x1, y1, nx2, ny2, np.sqrt((nx2 - x1) ** 2 + (ny2 - y1) ** 2))
            else:
                merged_lines.append(current_line)
                current_line = next_line

        merged_lines.append(current_line)
        return merged_lines

    merged_lines = merge_lines(horizontal_lines)

    # Sort merged lines by length and select the two largest
    merged_lines.sort(key=lambda x: x[4], reverse=True)
    largest_lines = merged_lines[:2]

    if not largest_lines:
        return None, "No horizontal lines found"

    # Draw the lines on the image for visualization
    for line in largest_lines:
        x1, y1, x2, y2, _ = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save or display the result image with detected lines
    result_image_path = 'output.png'
    cv2.imwrite(result_image_path, image)

    return largest_lines, result_image_path


# Usage example
image_path = 'DSC03044.JPG'  # Path to your image

lines, result_image = find_largest_horizontal_lines(image_path)
if lines:
    print("Detected Lines:", lines)
    print("Result Image Saved at:", result_image)
else:
    print("Error:", result_image)