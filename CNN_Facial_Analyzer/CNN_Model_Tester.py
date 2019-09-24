import numpy as np
import pandas as pd
import keras as K
import cv2
import CNN_Dataset
import matplotlib.pyplot as plt
from keras.models import load_model

emoList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Suprise']

data = CNN_Dataset.DATA()
model = load_model('CNN_Facial_Sensitivity_Analyzer_82%.h5')

if __name__ == "__main__":
    verifyCnt, datasetLen = 0, len(data.x_test)
    predict = model.predict(data.x_test)
    real = data.y_test
    
    for i in range(datasetLen):
        print('{}. predict: {}, real = {}'.format(i + 1, emoList[predict], emoList[real]))

        if predict == real:
            verifyCnt += 1                                                                                                                          
    
    print('Accuracy: {}'.format(verifyCnt / datasetLen))