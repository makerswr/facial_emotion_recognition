import keras as K
from keras import models
from keras.layers import Conv2D, Maxpooling2D
from keras.layers import Dense, Dropout, BatchNormalization, Flatten

class Model(models.Sequential):
    def __init__(self, inputShape, numOfClass):
        super().__init__()
        # 2D convolution layer 1
        self.add(layers.Conv2D(32, kernel_size = (3, 3),
                                   activation = 'relu',
                                   input_shape = inputShape))
        self.add(layers.Conv2D(32, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(layers.Conv2D(32, kernel_size = (3, 3), 
                                   activation = 'relu'))                          
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.25))

        # 2D convolution layer 2
        self.add(layers.Conv2D(64, kernel_size = (3, 3),
                                   activation = 'relu',))
        self.add(layers.Conv2D(64, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(layers.Conv2D(64, kernel_size = (3, 3), 
                                   activation = 'relu'))                           
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.25))

        # 2D convolution layer 3
        self.add(layers.Conv2D(128, kernel_size = (3, 3),
                                   activation = 'relu',))
        self.add(layers.Conv2D(128, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(layers.Conv2D(128, kernel_size = (3, 3), 
                                   activation = 'relu'))                           
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.25))

        # 2D convolution layer 4
        self.add(layers.Conv2D(256, kernel_size = (3, 3),
                                   activation = 'relu',))
        self.add(layers.Conv2D(256, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(layers.Conv2D(256, kernel_size = (3, 3), 
                                   activation = 'relu'))                           
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.25))

        # 2D convolution layer 5
        self.add(layers.Conv2D(512, kernel_size = (3, 3),
                                   activation = 'relu',))
        self.add(layers.Conv2D(512, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(layers.Conv2D(512, kernel_size = (3, 3), 
                                   activation = 'relu'))                           
        self.add(layers.MaxPooling2D(pool_size = (2, 2)))
        self.add(layers.Dropout(0.2))

        self.add(layers.Flatten())

        # Dense Layer 1
        self.add(layers.Dense(4096, activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.2))

        # Dense Layer 2
        self.add(layers.Dense(4096), activation = 'relu')
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.15))

        # Dense Layer 3
        self.add(layers.Dense(4096, activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.15))

        # Dense Layer 4
        self.add(layers.Dense(4096, activation = 'relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.1))

        # Dense Output layer
        self.add(layers.Dense(numOfClass, activation = 'softmax'))

        self.compile(loss = K.losses.categorical_crossentropy,
                     optimizer = adam(learning_rate=0.00001, beta_1=0.85, beta_2=0.998, amsgrad=False),
                     metrics = ['accuracy'])
