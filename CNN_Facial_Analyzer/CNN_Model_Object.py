import keras as K
from keras import models
from keras.layers import Conv2D, Maxpooling2D
from keras.layers import Dense, Dropout, BatchNormalization, Flatten

class Model(models.Sequential):
    def __init__(self, inputShape, numOfClass):
        super().__init__()

        self.add(layers.Conv2D(32, kernel_size = (3, 3),
                                   activation = 'relu',
                                   input_shape = inputShape))
        self.add(layers.BatchNormalization())
        self.add(layers.Conv2D(32, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.5))

        self.add(layers.Conv2D(64, kernel_size = (3, 3),
                                   activation = 'relu',
                                   input_shape = inputShape))
        self.add(layers.BatchNormalization())
        self.add(layers.Conv2D(64, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.5))

        self.add(layers.Flatten())

        self.add(layers.Dense(128, activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.5))

        self.add(layers.Dense(256, activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.25))

        self.add(layers.Dense(512, activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.125))

        self.add(layers.Dense(1024, activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.1))

        self.add(layers.Dense(numOfClass, activation = 'softmax'))

        self.compile(loss = K.losses.categorical_crossentropy,
                     optimizer = 'adam',
                     metrics = ['accuracy'])