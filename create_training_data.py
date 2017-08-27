
# import classes and functions 

from PIL import Image
import matplotlib.pyplot as plot
import os,sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

 
# Training Data
pathTrainDataCar = 'TrainImages/car/'
pathTrainDataBuilding = 'TrainImages/building/'
pathTrainDataRoad = 'TrainImages/road/'
pathTrainDataRandom = 'TrainImages/random/'
pathTestData = 'TrainImages/validation/'

pathResizedTrainDataCar = 'ResizeTrainImages/car/'
pathResizedTrainDataBuilding = 'ResizeTrainImages/building/'
pathResizedTrainDataRoad = 'ResizeTrainImages/road/'
pathResizedTrainDataRandom = 'ResizeTrainImages/random/'
pathResizedTestData = 'ResizeTrainImages/validation/'

# Resize the images 
row,column = 100,100

listingCar = os.listdir(pathTrainDataCar)
print(listingCar)
for file in listingCar:
	if file != '.DS_Store' :
		img = Image.open(pathTrainDataCar + file)
		resizeImg = img.resize((row,column))
		gray = resizeImg.convert('L')
		gray.save(pathResizedTrainDataCar + file)
	


listingBuilding = os.listdir(pathTrainDataBuilding)
print(listingBuilding)
for file in listingBuilding:
	if file != '.DS_Store':
		img = Image.open(pathTrainDataBuilding + file)
		resizeImg = img.resize((row,column))
		gray = resizeImg.convert('L')
		gray.save(pathResizedTrainDataBuilding + file)
	


listingRoad = os.listdir(pathTrainDataRoad)
print(listingRoad)
for file in listingRoad:
	if file != '.DS_Store':
		img = Image.open(pathTrainDataRoad + file)
		resizeImg = img.resize((row,column))
		gray = resizeImg.convert('L')
		gray.save(pathResizedTrainDataRoad + file)
	


listingRandom = os.listdir(pathTrainDataRandom)
print(listingRandom)
for file in listingRandom:
	if file != '.DS_Store':
		img = Image.open(pathTrainDataRandom + file)
		resizeImg = img.resize((row,column))
		gray = resizeImg.convert('L')
		gray.save(pathResizedTrainDataRandom + file)


listingTestData = os.listdir(pathTestData)
print(listingTestData)
for file in listingTestData:
	if file != '.DS_Store':
		img = Image.open(pathTestData+file)
		resizeImage = img.resize((row,column))
		gray = resizeImage.convert('L')
		gray.save(pathResizedTestData+file)
	



