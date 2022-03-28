#NOTE: FOR 7TH LINE IN CODE "cv2.VideoCapture(IT MAY BE "0" OR "1") TO ACCES WEBCAM IT VARIES DEVICE TO DEVICE 

import cv2
import numpy as np

extracting_haarfeatures = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#extracting featoures from xml file to apply on image
capture = cv2.VideoCapture(1)

data = [] #empty data file to save collected images
while True:
    flag,img = capture.read() #reading images frame by frame and if flag is true it proceeds further
    if flag:
       face = extracting_haarfeatures.detectMultiScale(img)  #it returns position(x,y) ,width,height of detected faces 
    
       for x,y,w,h in face:
           cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #drawing rectangle over found faces
           face = img[y:y+h, x:x+w, :]   #slicing face from image
           face = cv2.resize(face,(50,50)) # differnt images/frames may have differnt face sizes so reducing to same size
           print(len(data))
           if len(data)<200: #storing faces in data variable
               data.append(face)

       cv2.imshow("face",img)
       if cv2.waitKey(2)==49 or len(data)>=200:
           break

cv2.destroyAllWindows()
print(len(data))

#saving data
#np.save("data_without_mask",data)
#np.save("data_with_mask",data)

