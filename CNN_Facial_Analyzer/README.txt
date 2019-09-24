CNN training 까지 완성.
Epochs는 직접 입력하고, 파일 경로에 workpath 지정해주고 컴파일 시키고,
다른 데이터 셋으로 비교 하려면, CNN_Dataset.py 파일에 numOfClass 를 분류 개수로 바꾸고, datasetCSV 부분에서 데이터셋 바꾸고, x_train, t_test, y_train, y_test 에서 28170 부분을 원하는 만큼 바꾸기
