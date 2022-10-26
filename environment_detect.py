# streaming yolov5 辨識有無安全帽

import math
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from IPython.display import clear_output
import requests


os.chdir("C:\\Users\\andy4\\yolov5")

# 載入自行訓練的 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path='E:\\helmet_detect\\all_helmet_data\\results\\head-2022-10-18-yolov5s-new\\head-2022-10-18-yolov5s-new.pt')

model2 = torch.load('Resnet18_98199.pt')
# 設定 IoU 門檻值
model.iou = 0.5
# 設定信心門檻值
model.conf = 0.5

# 影像來源
#img_path = "http://video.itri.go:1010/video/store" # 串流網址
img_path = 0 # 筆電鏡頭

cap = cv2.VideoCapture(img_path)

correct = 0
y_true = []
y_pred = []
model2.eval()

transform = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

flag_sum = 0


flag = 1

while(True):
    
    ret, img = cap.read()
    
    red_flag = 0
    
    cv2.imwrite('output_temp.png', img)
    
    # 進行物件偵測
    results = model('output_temp.png')
    
    # return the predictions as a pandas dataframe
    bbox_df = results.pandas().xyxy[0]

    img = cv2.imread('output_temp.png')
    
    sec_img = img.copy()

    for bbox_number in range(len(bbox_df)):
    
        # 偵測到的bounding box
        xmin = int(bbox_df['xmin'][bbox_number])
        ymin = int(bbox_df['ymin'][bbox_number])
        xmax = int(bbox_df['xmax'][bbox_number])
        ymax = int(bbox_df['ymax'][bbox_number])
        confidence = str(round(bbox_df['confidence'][bbox_number],2))

        crop_img = []
        crop_img = img[ymin:ymax, xmin:xmax].copy()
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, "head " + str(confidence), (xmin-5, ymin-5), cv2.FONT_HERSHEY_DUPLEX,0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 已經切好的暫存img
        cv2.imwrite('detected_results_temp.png', crop_img)
        
        img_2 = Image.open('detected_results_temp.png').convert('RGB')
        
        image_tensor = transform(img_2)
        images = image_tensor.unsqueeze(0)
        images = images.to(device)
        
        outputs = model2(images)
        
        _ = []
        
        _, predicted = torch.max(outputs.data, 1)
        y_pred = int(predicted.cpu().numpy())
        
        label_0 = 1/(1 + math.exp(float(outputs.squeeze().tolist()[0])))
        label_1 = 1/(1 + math.exp(float(outputs.squeeze().tolist()[1])))
        
        if label_0 > label_1:
            res_result = str(round((label_0),2))
        else:
            res_result = str(round((label_1),2))
        
        if y_pred == 1:
            cv2.rectangle(sec_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(sec_img, "Helmet " + str(res_result), (xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            red_flag = 1
            cv2.rectangle(sec_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(sec_img, "UnHelmet " + str(res_result), (xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(sec_img, (500,0), (640,30), (0,0,255), cv2.FILLED) # text background
            cv2.putText(sec_img, "Detected", (520, 25), cv2.FONT_HERSHEY_DUPLEX,0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        #print('predict: ', y_pred)
    
    if red_flag == 0:
        flag_sum = 0
    else:
        flag_sum = flag_sum + 1
        
    if flag_sum > 100:
        flag_sum = 0
        print("activate")
        # call API (send sec_img)
        if flag == 1:
            cv2.imwrite('hi.png', sec_img)
            url = "http://handler.tsmc.n0b.me/api/v1/alert"
            files = {'file': open('hi.png', 'rb')}
            response = requests.post(url, files=files)
            flag = 0
            try:
                print(response.text)           
            except requests.exceptions.RequestException:
                print(response.text)

        

        
    cv2.putText(img, "Yolo Result", (5, 30), cv2.FONT_HERSHEY_DUPLEX,0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(sec_img, "ResNet Result", (5, 30), cv2.FONT_HERSHEY_DUPLEX,0.8, (255, 0, 0), 1, cv2.LINE_AA)
    
        
    cv2.imshow('first_stage_frame', img)
    cv2.imshow('second stage frame',sec_img)

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
