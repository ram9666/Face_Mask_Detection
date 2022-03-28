#NOTE: FOR 42TH LINE IN CODE "cv2.VideoCapture(IT MAY BE "0" OR "1") TO ACCES WEBCAM IT VARIES DEVICE TO DEVICE

#importing libraries
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score  #to find accuracy of trained data with test dta
from sklearn.model_selection import train_test_split  # for spliting data to train and test 
from sklearn.decomposition import PCA   #to change deimensions

data_with_mask = np.load("data_with_mask.npy")  #importing and loading collected data
data_without_mask = np.load("data_without_mask.npy")
#print(data_with_mask.shape)

data_with_mask = data_with_mask.reshape(200,50*50*3) #reshaping data to same dimensions 
data_without_mask = data_without_mask.reshape(200,50*50*3)
print(data_without_mask.shape)

x = np.r_[data_with_mask,data_without_mask] #concatenating data horizontally
print(x.shape)

labels = np.zeros(x.shape[0]) #labelling
labels[200:] = 1.0
names = {0 : 'mask', 1 : 'no_mask'}

x_train, x_test, y_train, y_test = train_test_split(x,labels,test_size=0.25) #assigning some part of dataset to traing and testing
print(x_train.shape)

pca = PCA(n_components=3)  #dimensional change of data 2dim to 3dimension
x_train= pca.fit_transform(x_train)
print(x_train.shape)

#feeding data to algirith,
svm = SVC()
svm.fit(x_train,y_train)
x_test = pca.transform((x_test)) 
y_pred = svm.predict(x_test)
print(accuracy_score(y_test,y_pred)) 

# now algrothim is trained and time to predict mask
haardata_extract = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#extracting haar features
capture = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX #font fearture(like size or style) to write strings  on images
data = []
while True:
    flag,img = capture.read()
    if flag:
       faces = haardata_extract.detectMultiScale(img)
       for x,y,w,h in faces:
           cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
           face = img[y:y+h, x:x+w, :]
           face = cv2.resize(face,(50,50))
           face = face.reshape(1,-1)
           face = pca.transform(face)
           pred = svm.predict(face) #predicting as "0" or "1"

           n = names[int(pred)]
           if n=='no_mask':
             cv2.putText(img,n,(x+int(w/3),y),font,1,(0,255,0),2)# writing ttext on face
           elif n=='mask':
             cv2.putText(img,n,(x+int(w/3),y),font,1,(0,0,255),2)
           #print(n)

       cv2.imshow("face",img)
       if cv2.waitKey(2)==49:
           break
cv2.destroyAllWindows()

