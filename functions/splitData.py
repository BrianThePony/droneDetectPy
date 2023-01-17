#%%

import numpy as np
import random
import os
import shutil

random.seed(10)

vedbaek = os.listdir("../../Datasets/drone_vedbaek/images/")
vaerloese = os.listdir("../../Datasets/drone_vaerloese/images")
random.shuffle(vedbaek)
random.shuffle(vaerloese)

train_ved = vedbaek[:int(len(vedbaek)*0.9)]
train_vaer = vaerloese[:int(len(vaerloese)*0.9)]
val_ved = vedbaek[int(len(vedbaek)*0.9):]
val_vaer = vaerloese[int(len(vaerloese)*0.9):]

for i in train_ved:
    src_i = "../../Datasets/drone_vedbaek/images/"+i
    dst_i = "../../Datasets/Train/images/"+i
    src_l = "../../Datasets/drone_vedbaek/labels/"+i[:-4]+'.txt'
    dst_l = "../../Datasets/Train/labels/"+i[:-4]+'.txt'
    
    shutil.copy(src_i,dst_i)
    shutil.copy(src_l,dst_l)

for i in train_vaer:
    src_i = "../../Datasets/drone_vaerloese/images/"+i
    dst_i = "../../Datasets/Train/images/"+i
    src_l = "../../Datasets/drone_vaerloese/labels/"+i[:-4]+'.txt'
    dst_l = "../../Datasets/Train/labels/"+i[:-4]+'.txt'
    
    shutil.copy(src_i,dst_i)
    shutil.copy(src_l,dst_l)


for i in val_ved:
    src_i = "../../Datasets/drone_vedbaek/images/"+i
    dst_i = "../../Datasets/Val/images/"+i
    src_l = "../../Datasets/drone_vedbaek/labels/"+i[:-4]+'.txt'
    dst_l = "../../Datasets/Val/labels/"+i[:-4]+'.txt'

    shutil.copy(src_i,dst_i)
    shutil.copy(src_l,dst_l)
for i in val_vaer:
    src_i = "../../Datasets/drone_vaerloese/images/"+i
    dst_i = "../../Datasets/Val/images/"+i
    src_l = "../../Datasets/drone_vaerloese/labels/"+i[:-4]+'.txt'
    dst_l = "../../Datasets/Val/labels/"+i[:-4]+'.txt'

    shutil.copy(src_i,dst_i)
    shutil.copy(src_l,dst_l)
    

