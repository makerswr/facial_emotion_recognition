import numpy as np
import keras as K
import pandas as pd
from keras import backend
from keras import datasets
from matplotlib import pyplot as plt

ANGRY, DISGUST, FEAR, HAPPY, SAD, SURPRISE, NEUTRAL = 0, 1, 2, 3, 4, 5, 6

class DATA():
    def __init__(self):
        numOfClass = 7

        datasetCSV = pd.read_csv('fer2013.csv', header = None)
        pixels = datasetCSV.values[:, 1]
        y = datasetCSV.values[:, 0]

        X = np.zeros((pixels.shape[0], 48*48))

        for i in range(X.shape[0]):
            p = pixels[i].split(' ')

            for j in range(X.shape[1]):
                X[i, j] = int(p[j])

        x_train = X[0:28710, :]
        y_train = y[0:28710]
        x_test = X[28710:32300, :]
        y_test = y[28710:32300]
        
        imgRows, imgCols = 48, 48

        backend.set_image_data_format('channels_last')
        x_train = x_train.reshape(x_train.shape[0], imgRows, imgCols, 1)
        x_test = x_test.reshape(x_test.shape[0], imgRows, imgCols, 1)
        inputShape = (imgRows, imgCols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = K.utils.to_categorical(y_train, numOfClass)
        y_test = K.utils.to_categorical(y_test, numOfClass)

        self.inputShape = inputShape
        self.numOfClass = numOfClass
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test