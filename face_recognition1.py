import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import cv2
import os
import imutils
import argparse


argp=argparse.ArgumentParser()
argp.add_argument('-p','--protxt',required=True,help='Protxt file for model architecture')
argp.add_argument('-m','--model',required=True,help='saved model file')
argp.add_argument('-c','--confidence',default=0.5,help='confidence of detected face',type=float)
arguments=vars(argp.parse_args())



knn = KNeighborsClassifier(n_neighbors=4)

cap = cv2.VideoCapture(0)

labels = []
path = 'data/'
name={}
index = 0
face_data = []

for fx in os.listdir(path):

	if fx.endswith('.npy'):
		name[index] = fx[:-4]
		file = np.load(path+fx,allow_pickle = True)
		face_data.append(file)
		labels.append(index*np.ones((file.shape[0],)))
		index = index+1

face_train = np.concatenate(face_data,axis = 0)
labels = np.concatenate(labels,axis = 0).reshape((-1,1))
train_data = np.concatenate((face_train,labels),axis =1)

knn.fit(train_data[:,:-1],train_data[:,-1])

cnet = cv2.dnn.readNetFromCaffe(arguments["protxt"], arguments["model"])

while True:

	ret,frame=cap.read()
	if ret==False:
		continue

	#cv2.imshow('frame',frame)
	
	frame = cv2.flip(frame,1)
	frame = imutils.resize(frame, width=400)

	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))# frame,scalefactor,size,mean
	
	(h, w) = frame.shape[:2]
	cnet.setInput(blob)
	predictions=cnet.forward()


	
	for i in range(predictions.shape[2]): #predictions.shape[2] is the number of detected objects

		confidence=predictions[0,0,i,2]

		if confidence<arguments['confidence']:
			continue

		#text = "{:.4f}%".format(confidence * 100)


		box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
		
		(startX, startY, endX, endY) = box.astype("int")

		

		#print(startX,startY)
		
		if startY<0:
			startY = 0
		if startX<0:
			startX = 0

		a = frame[ startY : endY , startX : endX ]
		a = cv2.resize(a,(100,100))
		a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
		a = np.array(a)
		a = a.reshape(1,-1)
		text = name[int(knn.predict(a))]
		offset=10
		y=startY-offset
		cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 2)
		cv2.rectangle(frame,(startX,startY),(endX,endY),(25,102,205),3)
		cv2.imshow('face',frame)

		#forcc = cv2.VideoWriter_fourcc(*'YUY2')
		#video = cv2.VideoWriter('data/video.mp4',forcc,20.0,(640,480))
	

	if cv2.waitKey(1) & 0xFF == ord('q') :
		break
	
cap.release()
cv2.destroyAllWindows()




#print(train_data[:,:-1])
#print(train_data[:,-1])


#for i in range(1000):
#	for j in range(1000):
#		if train_data[i][j] != 255:
#			print(train_data[i][j])