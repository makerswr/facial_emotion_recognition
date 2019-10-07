import keras as K
from keras import layers, models

def Model(inputShape, numOfClass):
    model = models.Sequential(inputShape, numOfClass)

    model.add(layers.Conv2D(32, kernel_size = (3, 3),
                                activation = 'relu',
                                input_shape = inputShape))
    model.add(layers.Conv2D(64, kernel_size = (3, 3), 
                                activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(256, activation = 'relu'))
    # model.add(layers.Dropout(0.3))
    model.add(layers.Dense(numOfClass, activation = 'softmax'))

    model.compile(loss = K.losses.categorical_crossentropy,
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])

    return model
