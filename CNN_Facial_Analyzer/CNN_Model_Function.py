import keras as K
from keras import layers, models

def Model(inputShape, numOfClass):
    # 2D convolution layer 1
    self.add(layers.Conv2D(32, kernel_size = (3, 3),
                               activation = 'relu',
                               input_shape = inputShape))
    self.add(layers.Conv2D(32, kernel_size = (3, 3), 
                               activation = 'relu'))
    self.add(layers.Conv2D(32, kernel_size = (3, 3), 
                               activation = 'relu'))                          
    self.add(layers.MaxPooling2D(pool_size = (2, 2)))
    self.add(layers.Dropout(0.5))

    # 2D convolution layer 2
    self.add(layers.Conv2D(64, kernel_size = (3, 3),
                               activation = 'relu',)
    self.add(layers.Conv2D(64, kernel_size = (3, 3), 
                               activation = 'relu'))
    self.add(layers.Conv2D(64, kernel_size = (3, 3), 
                               activation = 'relu'))                           
    self.add(layers.MaxPooling2D(pool_size = (2, 2)))
    self.add(layers.Dropout(0.5))
    
    # 2D convolution layer 3
    self.add(layers.Conv2D(128, kernel_size = (3, 3),
                               activation = 'relu',)
    self.add(layers.Conv2D(128, kernel_size = (3, 3), 
                          activation = 'relu'))
    self.add(layers.Conv2D(128, kernel_size = (3, 3), 
                               activation = 'relu'))                           
    self.add(layers.MaxPooling2D(pool_size = (2, 2)))
    self.add(layers.Dropout(0.5))

    # 2D convolution layer 4
    self.add(layers.Conv2D(256, kernel_size = (3, 3),
                               activation = 'relu',)
    self.add(layers.Conv2D(256, kernel_size = (3, 3), 
                               activation = 'relu'))
    self.add(layers.Conv2D(256, kernel_size = (3, 3), 
                               activation = 'relu'))                           
    self.add(layers.MaxPooling2D(pool_size = (2, 2)))
    self.add(layers.Dropout(0.5))

    # 2D convolution layer 5
    self.add(layers.Conv2D(512, kernel_size = (3, 3),
                               activation = 'relu',)
    self.add(layers.Conv2D(512, kernel_size = (3, 3), 
                               activation = 'relu'))
    self.add(layers.Conv2D(512, kernel_size = (3, 3), 
                               activation = 'relu'))                           
    self.add(layers.MaxPooling2D(pool_size = (2, 2)))
    self.add(layers.Dropout(0.5))
    
    self.add(layers.Flatten())

    # Dense Layer 1
    self.add(layers.Dense(4096, activation = 'relu'))
    self.add(layers.BatchNormalization())
    self.add(layers.Dropout(0.5))
   
    # Dense Layer 2
    self.add(layers.Dense(4096, activation = 'relu'))
    self.add(layers.BatchNormalization())
    self.add(layers.Dropout(0.25))

    # Dense Layer 3
    self.add(layers.Dense(4096, activation = 'relu'))
    self.add(layers.BatchNormalization())
    self.add(layers.Dropout(0.125))

    # Dense Layer 4
    self.add(layers.Dense(4096, activation = 'relu'))
    self.add(layers.BatchNormalization())
    self.add(layers.Dropout(0.1))

    # Dense Output layer
    self.add(layers.Dense(numOfClass, activation = 'softmax'))

    self.compile(loss = K.losses.categorical_crossentropy,
                 optimizer = 'adam',
                 metrics = ['accuracy'])

    return model
