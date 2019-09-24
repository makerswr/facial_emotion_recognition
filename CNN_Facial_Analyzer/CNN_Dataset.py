import numpy as np
import keras as K
import pandas as pd
from keras import backend
from keras import datasets

ANGRY, DISGUST, FEAR, HAPPY, SAD, SURPRISE, NEUTRAL = 0, 1, 2, 3, 4, 5, 6

class DATA():
    def __init__(self):
        numOfClass = 7

        datasetCSV = pd.read_csv('fer2013.csv', header = None)
        x_datasetList = []
        y_datasetList = []
        lenOfDataset = len(datasetCSV[1])

        for i in range(len(datasetCSV[1])):
            x_datasetList.append([datasetCSV[1][i]])
            y_datasetList.append(datasetCSV[0][i])

        datasetList = np.array(datasetList)

        (x_train, y_train), (x_test, y_test) = \
        (x_datasetList[:len(datasetCSV[1]) * (4 / 5)], y_datasetList[:len(datasetCSV[1]) * (4 / 5)]),
        (x_datasetList[:len(datasetCSV[1]) * (1 / 5)], y_datasetList[:len(datasetCSV[1]) * (1 / 5)])
        
        imgRows, imgCols = x_train.shape[1:]

        backend.image_data_format('channels_last')
   
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