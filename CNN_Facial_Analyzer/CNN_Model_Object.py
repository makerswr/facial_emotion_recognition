import keras as K
from keras import layers, models

class Model(models.Sequential):
    def __init__(self, inputShape, numOfClass):
        super().__init__()

        self.add(layers.Conv2D(32, kernel_size = (3, 3),
                                   activation = 'relu',
                                   input_shape = inputShape))
        self.add(layers.Conv2D(64, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation = 'relu'))
        self.add(layers.Dropout(0.5))
        # model.add(layers.Dense(256, activation = 'relu'))
        # model.add(layers.Dropout(0.3))
        self.add(layers.Dense(numOfClass, activation = 'softmax'))

        self.compile(loss = K.losses.categorical_crossentropy,
                     optimizer = 'rmsprop',
                     metrics = ['accuracy'])