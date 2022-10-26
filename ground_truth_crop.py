# 用 ground truth 切割圖片 並依照class分類儲存至各類資料夾
# class : head, helmet

import cv2
import os
from PIL import Image
from IPython.display import clear_output

img_path = "E:\\helmet_detect\\all_helmet_data\\all_img\\"
label_path = "E:\\helmet_detect\\helm\\2_label\\"
head_save_path = "E:\\cut_image\\head\\"
helmet_save_path = "E:\\cut_image\\helmet\\"

os.chdir(img_path)
all_img_path = os.listdir(os.curdir)

os.chdir(label_path)
all_label_path = os.listdir(os.curdir)


for img_counter in range(0,len(all_img_path)):
    
    print("進度: " + str(int(img_counter/len(all_img_path)*100)) + " %")
    
    text = []
    img = []
    orig_img = []
    
    img = cv2.imread(img_path + str(all_img_path[img_counter]))
    orig_img = Image.open(img_path + str(all_img_path[img_counter]))

    f = open(label_path + str(all_label_path[img_counter]),'r')

    for line in f.readlines(): # each row
        # 1 0.113 0.1746987951807229 0.078 0.1566265060240964
        text.append(line)
        
    f.close()

    line_store = []

    image_width = orig_img.width
    image_height = orig_img.height

    for line_counter in range(0, len(text)):
        line_store = text[line_counter].split()
    
        x_yolo = float(line_store[1])
        y_yolo = float(line_store[2])
        yolo_width = float(line_store[3])
        yolo_height = float(line_store[4])
    
        box_width = yolo_width * image_width
        box_height = yolo_height * image_height
    
        x_min = int(x_yolo * image_width - (box_width / 2))
        y_min = int(y_yolo * image_height - (box_height / 2))
        x_max = int(x_yolo * image_width + (box_width / 2))
        y_max = int(y_yolo * image_height + (box_height / 2))
    
        crop_img = []
        crop_img = img[y_min:y_max, x_min:x_max].copy()
        
        clear_output(wait=True)
    
        # without helmet
        if line_store[0] == "0":
            cv2.imwrite(head_save_path + str(all_img_path[img_counter][:-4]) + "_" + str(line_counter) + '.jpg', crop_img)   
        # with helemt
        else:
            cv2.imwrite(helmet_save_path + str(all_img_path[img_counter][:-4]) + "_" + str(line_counter) + '.jpg', crop_img) 
