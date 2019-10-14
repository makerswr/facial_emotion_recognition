import keras
from keras import models, layers
from keras import backend

class CNN(models.Sequential):
    def __init__(self, inputShape, numOfClass):
        super().__init__()

        self.add(layers.Conv2D(32, kernel_size = (3, 3),
                                   activation = 'relu',
                                   input_shape = inputShape))
        self.add(layers.Conv2D(32, kernel_size = (3, 3),
                                   activation = 'relu'))
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation = 'relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(numOfClass, activation = 'softmax'))

        self.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop',
                      metrics=['accuracy'])