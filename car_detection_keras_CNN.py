# import classes and functions 

from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D,Dropout, Flatten
from keras.utils import np_utils
from PIL import Image
import os,sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from array import array
import numpy as np 
import matplotlib.pyplot as plot
np.random.seed(0)  #for reproducibility 

 
# Training & Testing Data
row,column = 100,100
testCarExamples,testNonCarExamples = 122,116 # Documentation purpose 
pathResizedTrainDataCar = 'ResizeTrainImages/car/'
pathResizedTrainDataBuilding = 'ResizeTrainImages/building/'
pathResizedTrainDataRoad = 'ResizeTrainImages/road/'
pathResizedTrainDataRandom = 'ResizeTrainImages/random/'
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
	if file != '.DS_Store' :
		img = Image.open(pathResizedTrainDataCar + file)
		x = img_to_array(img)
		X_train.append(x)
		Y_train.append(0) # CAR 
	


print(X_train[0].shape)
print("Car Train data : %d" %len(X_train))   

# ........................... Building ...............................

listingBuilding = os.listdir(pathResizedTrainDataBuilding)
for file in listingBuilding:
	if file != '.DS_Store' :
		img = Image.open(pathResizedTrainDataBuilding + file)
		x = img_to_array(img)
		X_train.append(x)
		Y_train.append(1) # NOT CAR (Building)
	
	

# ........................... Road ....................................

listingRoad = os.listdir(pathResizedTrainDataRoad)
for file in listingRoad:
	if file != '.DS_Store' :
		img = Image.open(pathResizedTrainDataRoad + file)
		x = img_to_array(img)
		X_train.append(x)
		Y_train.append(1) # NOT CAR (Road)
	


# .......................... Random .................................

listingRandom = os.listdir(pathResizedTrainDataRandom)
for file in listingRandom:
	if file != '.DS_Store' :
		img = Image.open(pathResizedTrainDataRandom + file)
		x = img_to_array(img)
		X_train.append(x)
		Y_train.append(1) # NOT CAR (Random)

# ........................... Test Data ...............................

listingTestData = os.listdir(pathResizedTestData)
for file in listingTestData:
	if file != '.DS_Store' :
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
X_train = X_train.reshape(total_input, row, column, 1) 
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
X_test = X_test.reshape(total_testData, row, column , 1) 
X_test = X_test.astype('float32')     
X_test /= 255 
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(total_testData, 1)  


# It's a binary-class problem, output is 0 (CAR) and 1(NOT CAR). it's a good practice to use "one hot encoding" to class values 

print(Y_train.shape)
print(Y_train[0])

Y_train = np_utils.to_categorical(Y_train, classes) 
Y_test = np_utils.to_categorical(Y_test, classes)    
 
# Set up parameters
input_size = row * column
batch_size = 10    
hidden_neurons = 30    
epochs = 25
 

# Build the model
model = Sequential() 
model.add(Convolution2D(32, (2, 2), input_shape=(row, column, 1))) # 32 convolutional filter with size (2,2)
model.add(Activation('relu'))
model.add(Convolution2D(32, (2, 2)))  
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # filter size (2,2)
model.add(Dropout(0.5))  # Drop out is used for avoiding data overfitting by reducing the NN branches.              
model.add(Flatten())
  
model.add(Dense(hidden_neurons)) 
model.add(Activation('relu'))      
model.add(Dense(classes)) 
model.add(Activation('softmax'))


# Define Loss & compile model 
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adadelta') 
# optimizer - "rmsprop"/"sgd"/"adadelta" , loss - "binary_crossentropy" / "categorical_crossentropy"

# fit the model 

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, verbose=1)
 
# Test 
score = model.evaluate(X_test, Y_test, verbose=1)
print('\n''Test accuracy:', score[1]) 

# save model to create .mlmodel 
model.save('car_detection_keras_CNN_model.h5')






