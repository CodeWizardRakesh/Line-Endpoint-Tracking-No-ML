import cv2
import numpy as np

def detect_endpoints(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Edge detection
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)

    # Line detection using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=5)

    endpoints = []  # List to store endpoints
    intersection_points = set()  # Set to store intersection points

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            endpoints.append(((x1, y1), (x2, y2)))

            # Check for intersections (naive pairwise check)
            for other_line in lines:
                if np.array_equal(line, other_line):
                    continue
                x3, y3, x4, y4 = other_line[0]
                inter_point = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                if inter_point:
                    intersection_points.add(inter_point)

    return endpoints, list(intersection_points)

def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """Finds the intersection point of two line segments."""
    def det(a, b, c, d):
        return a * d - b * c

    denom = det(x1 - x2, y1 - y2, x3 - x4, y3 - y4)
    if denom == 0:
        return None  # Lines are parallel or coincident

    px = det(det(x1, y1, x2, y2), x1 - x2, det(x3, y3, x4, y4), x3 - x4) / denom
    py = det(det(x1, y1, x2, y2), y1 - y2, det(x3, y3, x4, y4), y3 - y4) / denom

    if is_between(x1, y1, x2, y2, px, py) and is_between(x3, y3, x4, y4, px, py):
        return int(px), int(py)
    return None

def is_between(x1, y1, x2, y2, px, py):
    """Checks if point (px, py) is on the line segment between (x1, y1) and (x2, y2)."""
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

# Example usage
image_path = 'D:\Freelance\End_track\line_img.jpg'
endpoints, intersections = detect_endpoints(image_path)
print("Endpoints:", endpoints)
print("Intersections:", intersections)
