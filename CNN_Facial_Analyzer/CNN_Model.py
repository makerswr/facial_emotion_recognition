import keras as K
from keras import models
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam

def Model(inputShape, numOfClass):
    model = models.Sequential()

    model.add(Conv2D(32, input_shape = inputShape,
                         kernel_size = (3, 3),
                         activation = 'relu'))
    model.add(Conv2D(32, kernel_size = (3, 3),
                         activation = 'relu',
                         padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, input_shape = inputShape,
                         kernel_size = (3, 3),
                         activation = 'relu'))
    model.add(Conv2D(64, kernel_size = (3, 3),
                         activation = 'relu',
                         padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, input_shape = inputShape,
                          kernel_size = (3, 3),
                          activation = 'relu'))
    model.add(Conv2D(128, kernel_size = (3, 3),
                          activation = 'relu',
                          padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.5))  

    model.add(Conv2D(256, input_shape = inputShape,
                          kernel_size = (3, 3),
                          activation = 'relu'))
    model.add(Conv2D(256, kernel_size = (3, 3),
                          activation = 'relu',
                          padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(numOfClass, activation = 'softmax'))

    model.compile(loss = K.losses.categorical_crossentropy,
                  optimizer = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                  metrics = ['accuracy'])
    
    return model