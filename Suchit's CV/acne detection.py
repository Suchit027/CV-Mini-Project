import cv2
import numpy as np

# reading the image
image = cv2.imread('4.jpg', 0)
height, width = image.shape[: 2]

# applying the min filter to remove white spots as much as possible
filtered = cv2.erode(image, np.ones((5, 5)), iterations=1)

edges = cv2.Canny(filtered, 30, 170)
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(filtered, contours, -1, (0, 255, 0), 2)

cv2.imshow('a', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
