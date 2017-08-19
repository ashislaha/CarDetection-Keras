import coremltools

DNN_ml_model = coremltools.converters.keras.convert('car_detection_keras_DNN_model.h5')
DNN_ml_model.author = 'Ashis Laha'
DNN_ml_model.description = 'Use for Car Detection'
DNN_ml_model.save('car_detection_keras_DNN.mlmodel')
print(DNN_ml_model)
