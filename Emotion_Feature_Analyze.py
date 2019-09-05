import numpy as np

NUMBER_FACIAL_LANDMARKS = 68 # number of facial landmarks

directory = './폴수학학교/2019 가을학기/전공 연구/Emotion_Feature_Analyze/dataset/'

def LoadDataset(emotion):
    dataset = list()
    path = directory + emotion + '.npy'
    datasetTemp = np.load(path, allow_pickle = True) # 3 dimension array : (x, y)
    
    for i in range(len(datasetTemp)): # index of face
        if datasetTemp[i]:
            dataset.append(datasetTemp[i])
            
    return np.array(dataset)

def AnalyzeDataset(dataset, dataCount):
    avgList = np.zeros((NUMBER_FACIAL_LANDMARKS, 2))
    xSum = list(0 for _ in range(NUMBER_FACIAL_LANDMARKS))
    ySum = list(0 for _ in range(NUMBER_FACIAL_LANDMARKS))

    for i in range(dataCount): # index of face
        for j in range(NUMBER_FACIAL_LANDMARKS): # index of facial landmark          
            xSum[j] += float(dataset[i][j][1])
            ySum[j] += float(dataset[i][j][2])

    for i in range(NUMBER_FACIAL_LANDMARKS): # index of facial landmark
        avgList[i][0] = xSum[i] / dataCount
        avgList[i][1] = ySum[i] / dataCount

    return avgList

def ExportAnalyzedDataset(emotion, avgList, dataCount):
    path = directory + emotion + 'AnalyzedData.txt'
    datasetFile = open(path, 'w')
    
    for i in range(NUMBER_FACIAL_LANDMARKS): # index of facial landmark
        datasetFile.write('%f %f\n' % (avgList[i][0], avgList[i][1]))

    datasetFile.close()

emotion = input('emotion: ')  
dataset = LoadDataset(emotion)
avgList = AnalyzeDataset(dataset, len(dataset))
ExportAnalyzedDataset(emotion, avgList, len(dataset))