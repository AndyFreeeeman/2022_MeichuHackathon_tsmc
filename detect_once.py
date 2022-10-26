# streaming yolov5 辨識有無安全帽

import math
import time
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
import socket
import threading
import requests

class SocketServer:
    # 建構式
    def __init__(self, host, port):
        # Socket Setting
        #HOST = '127.0.0.1'
        #PORT = 8000
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((host, port))
        self.signal = False
        print('server start at: %s:%s' % (host, port))
        print('wait for connection...')
    def listen(self):
        while not self.signal:
            global socket_data
            global outdata
            indata, addr = self.s.recvfrom(1024)
            print('recvfrom ' + str(addr) + ': ' + indata.decode())
            outdata = ''
            socket_data = indata.decode()
            
            while outdata == '':
                print(f'{outdata = }')
                time.sleep(0.5)
                pass
            
            self.s.sendto(outdata.encode(), addr)
    def exit(self):
        self.s.close()
        self.signal = True
        print("Close Socket")
        

socket_data = ""    
outdata = ''
socketServer = SocketServer("0.0.0.0", 8026)
t = threading.Thread(target=socketServer.listen)
t.start()


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

a = 0

while(True): # socket.listen
    if 1>0: # receive socket
        
        #if socket_data != "":
        #     a = 1
        
        safe_counter = 0
        for ten in range(10):
            ret, img = cap.read()

            cv2.imwrite('output_temp.png', img)

            # 進行物件偵測
            results = model('output_temp.png')

            # return the predictions as a pandas dataframe
            bbox_df = results.pandas().xyxy[0]

            img = cv2.imread('output_temp.png')

            max_area = 0

            main_xmin = 0
            main_xmax = 0
            main_ymin = 0
            main_ymax = 0

            main_crop_img = np.zeros((10,10),dtype=int)

            for bbox_number in range(len(bbox_df)):

                # 偵測到的bounding box
                xmin = int(bbox_df['xmin'][bbox_number])
                ymin = int(bbox_df['ymin'][bbox_number])
                xmax = int(bbox_df['xmax'][bbox_number])
                ymax = int(bbox_df['ymax'][bbox_number])
                confidence = str(round(bbox_df['confidence'][bbox_number],2))

                if ( (xmax-xmin)*(ymax-ymin) ) > max_area:
                    max_area = (xmax-xmin)*(ymax-ymin)
                    main_xmin = xmin
                    main_xmax = xmax
                    main_ymin = ymin
                    main_ymax = ymax

            main_crop_img = img[main_ymin:main_ymax, main_xmin:main_xmax].copy()
            # 已經切好的暫存img
            cv2.imwrite('detected_results_temp.png', main_crop_img)

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
                safe_counter = safe_counter + 1
                cv2.rectangle(img, (main_xmin, main_ymin), (main_xmax, main_ymax), (0, 255, 0), 1)
                cv2.putText(img, "Helmet " + str(res_result), (main_xmin-5, main_ymin-5), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (main_xmin, main_ymin), (main_xmax, main_ymax), (0, 0, 255), 2)
                cv2.putText(img, "UnHelmet " + str(res_result), (main_xmin-5, main_ymin-5), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(img, (505,0), (640,30), (0,0,255), cv2.FILLED) # text background
                cv2.putText(img, "Detected", (520, 25), cv2.FONT_HERSHEY_DUPLEX,0.8, (255, 255, 255), 1, cv2.LINE_AA)

            #cv2.putText(fir_img, "Yolo Result", (5, 30), cv2.FONT_HERSHEY_DUPLEX,0.8, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "ResNet Result", (5, 30), cv2.FONT_HERSHEY_DUPLEX,0.8, (255, 0, 0), 1, cv2.LINE_AA)


            #cv2.imshow('first_stage_frame', img)
            cv2.imshow('second stage frame',img)
            #cv2.imshow('main_detect', main_crop_img)

            # 若按下 q 鍵則離開迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                a=1
                
        if safe_counter < 5:
            outdata = 'notpass'
            
            if not socket_data == '':
                cv2.imwrite('hi.png', img)
                url = f"http://handler.tsmc.n0b.me/api/v1/alert?rfid={socket_data}"
                files = {'file': open('hi.png', 'rb')}
                response = requests.post(url, files=files)
                try:
                    print(response.text)           
                except requests.exceptions.RequestException:
                    print(response.text)
                    
            socket_data = ''
            
        else:
            outdata = 'pass'
            socket_data = ''
            # 8
            
        if a == 1:
            socketServer.exit()
            break
# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
