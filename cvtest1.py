import cv2

image=cv2.imread("dog.jpeg")

crop=image[400:600,200:400]

cv2.imshow("crop",crop)
cv2.waitKey()

