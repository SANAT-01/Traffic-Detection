import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import pandas as pd
from PIL import ImageGrab

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# def captureScreen(bbox=(0,0,960,1080)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

def labels(file):
    df = pd.read_csv(file)
    return df

vid = cv2.VideoCapture(0)
pickle_in = open("Best_model.p", "rb")  ## rb = READ BYTE
model = pickle.load(pickle_in)

df = labels('labels.csv')

while(True):
	ret, frame = vid.read()
	cv2.imshow('frame', frame)
	frame = grayscale(frame)
	frame = cv2.resize(frame, (32, 32))
	frame = frame.reshape(1, frame.shape[0], frame.shape[1], 1)
	#print(frame.shape)
	pred = model.predict([frame])
	classes_x = np.argmax(pred, axis=1)
	probabilityValue = np.amax(pred)
	
	#print(classes_x, probabilityValue)
	if probabilityValue>0.5:
		print(df.iloc[classes_x[0]]['Name'])
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()
plt.imshow(frame, 'gray')
plt.show()
