import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import pandas as pd

kaggle_csv = pd.read_csv('fer2013.csv', header = None)
print (kaggle_csv)
kgc = []
for i in range(len(kaggle_csv[1])):
    temp = kaggle_csv[1][i]
    temp = [int(j) for j in temp.split()]
    temp = np.array(temp, dtype = np.int32)
    temp = temp.reshape(48,48,1)
    kgc.append(temp)
    img = kgc[i].reshape(48,48)
    img = img.astype(np.uint8)
    result = Image.fromarray(img)
    result.save(str(i)+".jpg")
