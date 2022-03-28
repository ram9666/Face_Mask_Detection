import cv2
import numpy as np
img = cv2.imread("test_image.jpg")
extracting_haarfeatures = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face = extracting_haarfeatures.detectMultiScale(img)
for x,y,w,h in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
while True:
      cv2.imshow("face",img)
      if cv2.waitKey(2)==49:
           break
cv2.destroyAllWindows()