import CNN_Model
import CNN_Dataset
import matplotlib.pyplot as plt
from skeras import plot_acc, plot_loss
from keras.models import load_model

data = CNN_Dataset.DATA()
model = CNN_Model.CNN(data.inputShape, data.numOfClass)

if __name__ == "__main__":
    batchSize = 128
    epochs = int(input('Epochs: '))

    history = model.fit(data.x_train, data.y_train,
                        batch_size = batchSize,
                        epochs = epochs,
                        validation_split = 0.2)
              
    score = model.evaluate(data.x_train, data.y_train)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()

    model.save('CNN_Facial_Sensitivity_Analyzer.h5')