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
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=7) #Aperture size is always the odd numbers between 3 and 7

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=50)

    if lines is None:
        print("No lines detected.")
        return

    # Copy the original image for displaying results
    output_image = image.copy()

    # List to store endpoints
    endpoints = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append(((x1, y1), (x2, y2)))

        # Draw the detected line
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines

        # Mark the endpoints
        cv2.circle(output_image, (x1, y1), 5, (0, 0, 255), -1)  # Red circle at one endpoint
        cv2.circle(output_image, (x2, y2), 5, (255, 0, 0), -1)  # Blue circle at the other endpoint

    # Display the image with lines and endpoints marked
    cv2.imshow("Detected Endpoints", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print detected endpoints
    print("Detected Endpoints:", endpoints)

# Example usage
image_path = 'a_line.jpg'  # Replace with the path to your image
detect_and_display_endpoints(image_path)
