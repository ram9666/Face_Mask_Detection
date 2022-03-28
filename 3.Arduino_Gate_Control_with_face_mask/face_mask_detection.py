#NOTE: FOR 43TH LINE IN CODE "cv2.VideoCapture(IT MAY BE "0" OR "1") TO ACCES WEBCAM IT VARIES DEVICE TO DEVICE
#to run this arduino board should connect at 'COM4'
#importing libraries
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import Arduino_control

data_with_mask = np.load("data_with_mask.npy")  #importing and loading collected data
data_without_mask = np.load("data_without_mask.npy")
#print(data_with_mask.shape)
#print(data_without_mask.shape)

data_with_mask = data_with_mask.reshape(200,50*50*3) #reshaping data to same dimensions 
data_without_mask = data_without_mask.reshape(200,50*50*3)
print(data_without_mask.shape)

x = np.r_[data_with_mask,data_without_mask] #concatenating image
#labelling data set
labels = np.zeros(x.shape[0])
labels[200:] = 1.0
names = {0 : 'mask', 1 : 'no_mask'}

x_train, x_test, y_train, y_test = train_test_split(x,labels,test_size=0.20) #assigning some part of dataset to traing and testing
print(x_train.shape)

pca = PCA(n_components=3)
x_train= pca.fit_transform(x_train)
print(x_train.shape)

svm = SVC()
svm.fit(x_train,y_train)
x_test = pca.transform((x_test))
y_pred = svm.predict(x_test)
print(accuracy_score(y_test,y_pred)) 
# now algrothim is trained and time to predict mask

def face_mask():
    haardata_extract = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#extracting haar features
    capture_video = cv2.VideoCapture(1)
    font = cv2.FONT_HERSHEY_COMPLEX
    data = []
    data1 = [] 
    while True:
        flag,img = capture_video.read()
        if flag:
           faces = haardata_extract.detectMultiScale(img)
           for x,y,w,h in faces:
               cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
               face = img[y:y+h, x:x+w, :]
               face = cv2.resize(face,(50,50))
               face = face.reshape(1,-1)
               face = pca.transform(face)
               predict = svm.predict(face)
               n = names[int(predict)]
               if n=='no_mask':
                 cv2.putText(img,n,(x+int(w/3),y),font,1,(0,255,0),2)# writing ttext on face
               elif n=='mask':
                 cv2.putText(img,n,(x+int(w/3),y),font,1,(0,0,255),2)
             #print(n)
               if n=='mask':
                 data1.append(n)

        cv2.imshow("face",img)
        if cv2.waitKey(2)==49 or len(data1)==20:
            break
    cv2.destroyAllWindows()
    Arduino_control.led_motor(1)


#calling the function
face_mask() #calling above function
