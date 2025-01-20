import cv2
import numpy as np


image_path = 'D:\\Freelance\\Line-Endpoint-Tracking-No-ML\\two_lines.jpg'


image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold_value = 1
_, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)


print(gray_image)
print("Max pixel :",np.max(gray_image))
print("Min pixel :",np.min(gray_image))
cv2.imshow('Image', gray_image)
# cv2.imshow('Image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
