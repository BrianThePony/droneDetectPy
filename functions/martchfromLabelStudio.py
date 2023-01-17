#%%

import numpy as np
import os
dir = "../../datasets/"
vedbaek = os.listdir(dir+"/drone_vedbaek/images")
vaerloese = os.listdir(dir+"drone_vaerloese/images")

for i in range(len(vedbaek)):

    if len(vedbaek[i]) == 14:
        new_name = vedbaek[i][-5]+"_vedbaek"
    elif len(vedbaek[i]) == 15:
        new_name = vedbaek[i][-6:-4]+"_vedbaek"
    elif len(vedbaek[i]) == 16:
        new_name = vedbaek[i][-7:-4]+"_vedbaek"
    print(new_name)
    os.rename(dir+"/drone_vedbaek/images/"+vedbaek[i],dir+"/drone_vedbaek/images/"+new_name+".png")
    os.rename(dir+"/drone_vedbaek/labels/"+vedbaek[i][:-4]+".txt",dir+"/drone_vedbaek/labels/"+new_name+".txt")

for i in range(len(vaerloese)):

    if len(vaerloese[i]) == 14:
        new_name = vaerloese[i][-5]+"_vaerloese"
    elif len(vaerloese[i]) == 15:
        new_name = vaerloese[i][-6:-4]+"_vaerloese"
    elif len(vaerloese[i]) == 16:
        new_name = vaerloese[i][-7:-4]+"_vaerloese"

    os.rename(dir+"drone_vaerloese/images/"+vaerloese[i],dir+"drone_vaerloese/images/"+new_name+".png")
    os.rename(dir+"drone_vaerloese/labels/"+vaerloese[i][:-4]+".txt",dir+"drone_vaerloese/labels/"+new_name+".txt")
#%%

labelss = os.listdir(dir+"labels")


for i in labelss:
    if i.endswith(".png.txt"):
        os.rename(dir+"labels/"+i,dir+"labels/"+i[:-8]+'.txt')



