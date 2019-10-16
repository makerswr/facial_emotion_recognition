import keras as K
from keras import models
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam

class CNN(models.Sequential):
    def __init__(self, inputShape, numOfClass):
        super().__init__()
        # 2D convolution layer 1
        self.add(Conv2D(32, kernel_size = (3, 3),
                                   activation = 'relu',
                                   input_shape = inputShape))
        self.add(Conv2D(32, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(Conv2D(32, kernel_size = (3, 3), 
                                   activation = 'relu'))   
        self.add(Dropout(0.25))

        # 2D convolution layer 2
        self.add(Conv2D(64, kernel_size = (3, 3),
                                   activation = 'relu',))
        self.add(Conv2D(64, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(Conv2D(64, kernel_size = (3, 3), 
                                   activation = 'relu'))   
        self.add(Dropout(0.25))

        # 2D convolution layer 3
        self.add(Conv2D(256, kernel_size = (3, 3),
                                   activation = 'relu',))
        self.add(Conv2D(256, kernel_size = (3, 3), 
                                   activation = 'relu'))
        self.add(Conv2D(256, kernel_size = (3, 3), 
                                   activation = 'relu'))                           
        self.add(MaxPooling2D(pool_size = (2, 2)))
        self.add(Dropout(0.25))

        self.add(Flatten())

        # Dense Layer 1
        self.add(Dense(512, activation = 'relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.2))

        # Dense Layer 2
        self.add(Dense(4096, activation = 'relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.15))

        # Dense Layer 3
        self.add(Dense(4096, activation = 'relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.15))

        # Dense Layer 4
        self.add(Dense(4096, activation = 'relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.1))

        # Dense Output layer
        self.add(Dense(numOfClass, activation = 'softmax'))

        self.compile(loss = K.losses.categorical_crossentropy,
                     optimizer = Adam(lr=0.00001, beta_1=0.85, beta_2=0.998, amsgrad=False),
                     metrics = ['accuracy'])
