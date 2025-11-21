
import cv2
 
print("OpenCV 版本：", cv2.__version__)
print(cv2.getVersionString())
image=cv2.imread("/home/ddy/code/python/cvstudy/dog.jpeg")
print(image.shape)
#cv2.imshow("image",image)
#cv2.waitKey()


cv2.imshow("blue",image[:,:,0])
cv2.imshow("green",image[:,:,1])
cv2.imshow("red",image[:,:,2])

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey()