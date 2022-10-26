# 將dataset全部裝進train的images和labels
# 輸入valid_ratio即可將dataset切割為train跟valid

'''
home_path -----train----images
            |         ∟-labels
             ∟-valid----images(empty)
                      ∟-labels(empty)
'''


import os
import random
import shutil
from IPython.display import clear_output

home_path = "E:\\helmet_detect\\helm\\"

# 切valid的檔案數比例
valid_ratio = 0.1

os.chdir(home_path + "train\\images\\")
all_img_path = os.listdir(os.curdir)

all_number = []
random_store = []

for a in range(1, len(all_img_path)):
    all_number.append(a)
    
random_store = random.sample(all_number, int(len(all_img_path)*valid_ratio))
    
for random_counter in range(0,len(random_store)):
#for random_counter in range(0,1):
    
    chose_name = ""
    chose_name = all_img_path[random_store[random_counter]][:-4]
    
    shutil.move( home_path + "train\\images\\" + all_img_path[random_store[random_counter]] 
                ,home_path + "valid\\images\\" + all_img_path[random_store[random_counter]]  )
    shutil.move( home_path + "train\\labels\\" + str(chose_name) + ".txt" 
                ,home_path + "valid\\labels\\" + str(chose_name) + ".txt" )
    
    clear_output(wait=True)
    print("進度： " + str(int(random_counter/len(random_store)*100)+1) + " %" )
    print(all_img_path[random_store[random_counter]])
