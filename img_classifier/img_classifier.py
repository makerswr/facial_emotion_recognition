import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import pandas as pd
#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
kaggle_csv = pd.read_csv('fer2013.csv', header = None)
print (kaggle_csv)
kgc = []
for i in range(len(kaggle_csv[1])):
    if kaggle_csv[0][i] == 0:
        temp = kaggle_csv[1][i]
        temp = [int(j) for j in temp.split()]
        temp = np.array(temp, dtype = np.int32)
        temp = temp.reshape(48,48,1)
        kgc.append(temp)
        img = kgc[i].reshape(48,48)
        img = img.astype(np.uint8)
        result = Image.fromarray(img)
        result.save('./0/'+str(i)+".jpg")

    elif kaggle_csv[0][i] == 1:
        temp = kaggle_csv[1][i]
        temp = [int(j) for j in temp.split()]
        temp = np.array(temp, dtype = np.int32)
        temp = temp.reshape(48,48,1)
        kgc.append(temp)
        img = kgc[i].reshape(48,48)
        img = img.astype(np.uint8)
        result = Image.fromarray(img)
        result.save("./1/"+str(i)+".jpg")

    elif kaggle_csv[0][i] == 2:
        temp = kaggle_csv[1][i]
        temp = [int(j) for j in temp.split()]
        temp = np.array(temp, dtype = np.int32)
        temp = temp.reshape(48,48,1)
        kgc.append(temp)
        img = kgc[i].reshape(48,48)
        img = img.astype(np.uint8)
        result = Image.fromarray(img)
        result.save("./2/"+str(i)+".jpg")

    elif kaggle_csv[0][i] == 3:
        temp = kaggle_csv[1][i]
        temp = [int(j) for j in temp.split()]
        temp = np.array(temp, dtype = np.int32)
        temp = temp.reshape(48,48,1)
        kgc.append(temp)
        img = kgc[i].reshape(48,48)
        img = img.astype(np.uint8)
        result = Image.fromarray(img)
        result.save("./3/"+str(i)+".jpg")

    elif kaggle_csv[0][i] == 4:
        temp = kaggle_csv[1][i]
        temp = [int(j) for j in temp.split()]
        temp = np.array(temp, dtype = np.int32)
        temp = temp.reshape(48,48,1)
        kgc.append(temp)
        img = kgc[i].reshape(48,48)
        img = img.astype(np.uint8)
        result = Image.fromarray(img)
        result.save("./4/"+str(i)+".jpg")

    elif kaggle_csv[0][i] == 5:
        temp = kaggle_csv[1][i]
        temp = [int(j) for j in temp.split()]
        temp = np.array(temp, dtype = np.int32)
        temp = temp.reshape(48,48,1)
        kgc.append(temp)
        img = kgc[i].reshape(48,48)
        img = img.astype(np.uint8)
        result = Image.fromarray(img)
        result.save("./5/"+str(i)+".jpg")

    elif kaggle_csv[0][i] == 6:
        temp = kaggle_csv[1][i]
        temp = [int(j) for j in temp.split()]
        temp = np.array(temp, dtype = np.int32)
        temp = temp.reshape(48,48,1)
        kgc.append(temp)
        img = kgc[i].reshape(48,48)
        img = img.astype(np.uint8)
        result = Image.fromarray(img)
        result.save("./6/"+str(i)+".jpg")

    else:
        print("error")
