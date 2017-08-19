
# import classes and functions 

from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from PIL import Image
import matplotlib.pyplot as plot
import os,sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from array import array
import numpy as np 
import matplotlib.pyplot as plot
from random import shuffle

 
# Training & Testing Data
row,column = 100,100
testCarExamples,testNonCarExamples = 122,116 # Documentation purpose 
pathResizedTrainDataCar = 'ResizeTrainImages/car/'
pathResizedTrainDataBuilding = 'ResizeTrainImages/building/'
pathResizedTrainDataRoad = 'ResizeTrainImages/road/'
pathResizedTestData = 'ResizeTrainImages/validation/'

# create a matrix to flatten all the values

X_train = []
Y_train = []
X_test  = []
Y_test	= []
classes = 2 

# ............................ CAR ...................................

listingCar = os.listdir(pathResizedTrainDataCar)
for file in listingCar:
	img = Image.open(pathResizedTrainDataCar + file)
	x = img_to_array(img)
	X_train.append(x)
	Y_train.append(0) # CAR 


print(X_train[0].shape)
print("Car Train data : %d" %len(X_train))   

# ........................... Building ...............................

listingBuilding = os.listdir(pathResizedTrainDataBuilding)
for file in listingBuilding:
	img = Image.open(pathResizedTrainDataBuilding + file)
	x = img_to_array(img)
	X_train.append(x)
	Y_train.append(1) # NOT CAR (Building)
	

# ........................... Road ....................................

listingRoad = os.listdir(pathResizedTrainDataRoad)
for file in listingRoad:
	img = Image.open(pathResizedTrainDataRoad + file)
	x = img_to_array(img)
	X_train.append(x)
	Y_train.append(1) # NOT CAR (Road)


# ........................... Test Data ...............................

listingTestData = os.listdir(pathResizedTestData)
for file in listingTestData:
	img = Image.open(pathResizedTestData + file)
	x = img_to_array(img)
	X_test.append(x)
	if file.startswith("test_car"):
		Y_test.append(0) # CAR
	else:
		Y_test.append(1) # NOT CAR 


# ......................... Train Data Reshaping .......................
total_input = len(X_train)
print("Total Train Data : %d" %total_input)

X_train = np.array(X_train)
X_train = X_train.reshape(total_input, row*column) 
X_train = X_train.astype('float32')     
X_train /= 255 
Y_train = np.array(Y_train)   
Y_train = Y_train.reshape(total_input, 1)   

print("X_train shape")
print(X_train.shape)
print("Y_train Shape")
print(Y_train.shape)

# ................................. Plot Car in MatLab .....................

plot.subplot(221)
plot.imshow(X_train[0].reshape(row,column))
plot.show()

# ......................... Test Data Reshaping .......................

total_testData = len(X_test)
print("Total Test Data : %d" %total_testData)
X_test = np.array(X_test)
X_test = X_test.reshape(total_testData, row*column) 
X_test = X_test.astype('float32')     
X_test /= 255 
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(total_testData, 1)  


# It's a binary-class problem, output is 1 (CAR) and 2 (NOT CAR). it's a good practice to use "one hot encoding" to class values 

print(Y_train.shape)
print(Y_train[0])

Y_train = np_utils.to_categorical(Y_train, classes) 
Y_test = np_utils.to_categorical(Y_test, classes)    
 
# Set up parameters
input_size = row * column
batch_size = 10    
hidden_neurons = 100    
epochs = 50
 
# Build the model
model = Sequential()     
model.add(Dense(hidden_neurons, input_dim=input_size)) 
model.add(Activation('relu'))  
#model.add(Dropout(0.2))   
model.add(Dense(classes, input_dim=hidden_neurons)) 
model.add(Activation('softmax'))

# compile model 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

# fit the model 

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
 
# Test 
score = model.evaluate(X_test, Y_test, verbose=1)
print('\n''Test accuracy:', score[1]) 

# save model to create .mlmodel 
model.save('car_detection_keras_DNN_model.h5')

