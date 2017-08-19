from keras.models import load_model
import numpy as np 
import os,sys
from PIL import Image
import matplotlib.pyplot as plot
from optparse import OptionParser


model = load_model('car_detection_keras_DNN_model.h5')
row,column = 100,100

# take input to test it 
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", help="write report to FILE", metavar="FILE")
(options, args) = parser.parse_args()
URL = options.filename 
print("URL ------> %s" %URL)

img = Image.open(URL)
print("TESTING IMAGE -----> %s %s %s" %( img.bits, img.size, img.format))

# Resize the image
img = img.resize((row,column),Image.ANTIALIAS)

# Gray Scale Image 

def grayscale(picture):
    res= Image.new(picture.mode, picture.size)
    width, height = picture.size

    for i in range(0, width):
        for j in range(0, height):
            pixel=picture.getpixel((i,j))
            avg=(pixel[0]+pixel[1]+pixel[2])/3
            res.putpixel((i,j),(avg,avg,avg))
    res.show()
    return res

gray_image = grayscale(img)


# Normalize between 0 and 1 
def normalize(picture):
	width, height = picture.size
	normalized_array = []
	for j in range(0, height):
		for i in range(0, width):
			pixel = picture.getpixel((i,j))
			normalized_array.append( pixel[0] / 255.0 )
	return np.array(normalized_array)


X_test = normalize(gray_image)
X_test = X_test.reshape(1, row*column) 

# Do predictions 

classes = model.predict(X_test)
print(classes)

maxVal = classes[0].max()
indexVal = np.where(classes[0]==maxVal) # result iss an array


if (indexVal[0] == 0):
    print("......... It's CAR .........")
else: 
    print(".......... Not A Car .......")







