import sys
import os
import dlib
import glob
import cv2  #opencv 사용
import pandas as pd
import numpy as np
import copy
from scipy.interpolate import interp1d

def swapRGB2BGR(rgb):
    r, g, b = cv2.split(img)
    bgr = cv2.merge([b,g,r])
    return bgr
total = []
predictor_path = sys.argv[1] #얼굴 특징점 모델 파일 지정
faces_folder_path = sys.argv[2] #이미지 데이터 셋 path
csv_dataset = 'fer2013.csv' #데이터셋 파일 지정
emotion_classification = []
detector = dlib.get_frontal_face_detector() #얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
predictor = dlib.shape_predictor(predictor_path) #인식된 얼굴에서 랜드마크 찾기위한 클래스 생성

cv2.namedWindow('Face') #이미지를 화면에 표시하기 위한 openCV 윈도 생성
for f in glob.glob(os.path.join(faces_folder_path,"*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    cvImg = swapRGB2BGR(img) #불러온 이미지 데이터를 R과 B를 바꿔준다.
    cvImg = cv2.resize(cvImg, None, fx=2, fy=2, interpolation=cv2.INTER_AREA) #이미지를 두배로 키운다.
    dets = detector(img, 1) #업샘플링
    print("Number of faces detected: {}".format(len(dets))) #인식된 얼굴 개수 출력
    for k, d in enumerate(dets): # 얼굴 개수 만큼 윤곽 반복 출력, (K: 얼굴 인덱스, D: 얼굴 좌표)
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = predictor(img, d)  # 인식된 좌표에서 특징점 추출
        print(shape.num_parts)
        for i in range(0, shape.num_parts): # num_parts(특징점 구조체)를 하나씩 루프를 돌린다.
            x = shape.part(i).x*2
            y = shape.part(i).y*2
            print(str(x) + "," + str(y)) #특징점 좌표
            cv2.putText(cvImg, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0)) # 이미지 특징점 좌표 지점에 인덱스(랜드마크번호, 여기선 i)를 putText로 표시해준다.
            cv2.imshow('Face', cvImg)
            emotion_classification.append([i+1,str(x),str(y)])
            print(emotion_classification)
    total.append(copy.deepcopy(emotion_classification))
    emotion_classification = []

total = np.asarray(total)
np.save("data", total)
cv2.destroyWindow('Face')
