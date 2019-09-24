import CNN_Model
import CNN_Dataset
from keras.models import load_model

data = CNN_Dataset.DATA()
model = CNN_Model.CNN(data.inputShape, data.numOfClass)

if __name__ == "__main__":
    batchSize = 128
    epochs = int(input('Epochs: '))

    model.fit(data.x_train, data.y_train,
              batch_size = batchSize,
              epochs = epochs,
              validation_split = 0.2)
              
    score = model.evaluate(data.x_train, data.y_train)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save('CNN_Facial_Sensitivity_Analyzer.h5')
