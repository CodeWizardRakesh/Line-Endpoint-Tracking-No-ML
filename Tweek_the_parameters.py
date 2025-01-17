import cv2
import numpy as np

def detect_lines_with_parameters(image, canny_threshold1, canny_threshold2, hough_threshold, min_line_length, max_line_gap):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(gray_blurred, canny_threshold1, canny_threshold2)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    return lines

def experiment_with_parameters(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'. Check the file path.")
        return

    # Define parameter ranges to iterate through
    canny_thresholds = [(50, 150), (100, 200), (150, 250), (30, 80)]
    hough_thresholds = [50, 75, 100, 150]
    min_line_lengths = [30, 50, 70, 100]
    max_line_gaps = [50, 100, 150, 200]

    best_result = None

    # Iterate through parameter combinations
    for canny_threshold1, canny_threshold2 in canny_thresholds:
        for hough_threshold in hough_thresholds:
            for min_line_length in min_line_lengths:
                for max_line_gap in max_line_gaps:
                    # Detect lines with current parameters
                    lines = detect_lines_with_parameters(image, canny_threshold1, canny_threshold2, hough_threshold, min_line_length, max_line_gap)
                    
                    if lines is not None:
                        # Evaluate the number of lines detected
                        line_count = len(lines)
                        
                        # If exactly two lines are detected, store the result and break out
                        if line_count == 2:
                            best_result = lines
                            print(f"Found 2 lines with parameters: Canny({canny_threshold1}, {canny_threshold2}), Hough Threshold={hough_threshold}, Min Line Length={min_line_length}, Max Line Gap={max_line_gap}")
                            break
                if best_result is not None:
                    break
            if best_result is not None:
                break
        if best_result is not None:
            break

    # If the best result is found, draw the lines and print the coordinates
    if best_result is not None:
        output_image = image.copy()
        for line in best_result:
            x1, y1, x2, y2 = line[0]
            # Draw the detected line
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line
            # Mark the endpoints
            cv2.circle(output_image, (x1, y1), 5, (0, 0, 255), -1)  # Red circle at start point
            cv2.circle(output_image, (x2, y2), 5, (255, 0, 0), -1)  # Blue circle at end point

        # Display the result
        cv2.imshow("Detected Endpoints", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print detected lines
        print("Detected Endpoints for All Lines:")
        for i, line in enumerate(best_result, start=1):
            x1, y1, x2, y2 = line[0]
            print(f"Line {i}: Start = ({x1}, {y1}), End = ({x2}, {y2})")
    else:
        print("No suitable combination found for exactly 2 lines.")

# Example usage
image_path = 'two_lines.jpg'  # Replace with the path to your image
experiment_with_parameters(image_path)
