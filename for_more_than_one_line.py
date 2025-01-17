import cv2
import numpy as np

def detect_and_display_endpoints(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'. Check the file path.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Perform edge detection
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=7)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=200, maxLineGap=70)

    if lines is None:
        print("No lines detected.")
        return

    # Copy the original image for displaying results
    output_image = image.copy()

    # List to store endpoints for all lines
    all_endpoints = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        start_point = (x1, y1)
        end_point = (x2, y2)
        all_endpoints.append((start_point, end_point))

        # Draw the detected line
        cv2.line(output_image, start_point, end_point, (0, 255, 0), 2)  # Green line

        # Mark the endpoints
        cv2.circle(output_image, start_point, 5, (0, 0, 255), -1)  # Red circle at start point
        cv2.circle(output_image, end_point, 5, (255, 0, 0), -1)  # Blue circle at end point

    # Display the image with lines and endpoints marked
    cv2.imshow("Detected Endpoints", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the detected endpoints for all lines
    print("Detected Endpoints for All Lines:")
    for idx, endpoints in enumerate(all_endpoints, start=1):
        print(f"Line {idx}: Start = {endpoints[0]}, End = {endpoints[1]}")

# Example usage
image_path = 'two_lines.jpg'  # Replace with the path to your image
detect_and_display_endpoints(image_path)
