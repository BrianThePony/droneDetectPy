#%%
from openTif import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

fname = "../../DATA/VedbaekStrand_Ortho_noalpha.tif"

ds, bands = openTif(fname)

W = ds.RasterXSize
H = ds.RasterYSize
#plt.figure(figsize=(15,14))

#plt.imshow(bands['NIR'])

#plt.show()
### This is for Vaerloese
#bands['NIR'] = bands['NIR'][4000:8800,7000:11800]
#bands['R'] = bands['R'][4000:8800,7000:11800]
#bands['G'] = bands['G'][4000:8800,7000:11800]
#bands['B'] = bands['B'][4000:8800,7000:11800]
#bands['RE'] = bands['RE'][4000:8800,7000:11800]
###
### This is for Vedbaek
bands['NIR'] = bands['NIR'][1500:5200,1200:4400]
bands['R'] = bands['R'][1500:5200,1200:4400]
bands['G'] = bands['G'][1500:5200,1200:4400]
bands['B'] = bands['B'][1500:5200,1200:4400]
bands['RE'] = bands['RE'][1500:5200,1200:4400]

plt.figure(figsize=(15,14))

plt.imshow(bands['R'][int(640*0.35)*14:(int(640*0.35)*14)+640,int(640*0.35)*12:(int(640*0.35)*12)+640])


plt.show()

H,W = bands['NIR'].shape

#%%

w = 640
h = 640
s_x = np.floor(640*0.35)
s_y = np.floor(640*0.35)

num_im = (((H-h)//s_x)+1) * (((W-w)//s_y)+1)

Bands_smaller = []
Bands_smaller_vid = []
Bands_smaller_pca = []

BANDS = np.stack((bands['R']/np.max(bands['R']),bands['G']/np.max(bands['G']),bands['B']/np.max(bands['B']),bands['RE']/np.max(bands['RE']),bands['NIR']/np.max(bands['NIR'])),axis=-1)
BANDS_full = np.stack((bands['R'],bands['G'],bands['B'],bands['RE'],bands['NIR']),axis=-1)

#BANDS = BANDS/np.max(BANDS,axis=-1)

BANDS_video = np.stack((bands['R']/np.max(bands['R']) * 255, bands['G']/np.max(bands['G']) * 255,
                        bands['B']/np.max(bands['B'] * 255), bands['RE']/np.max(bands['RE']) * 255,
                        bands['NIR']/np.max(bands['NIR']) * 255),axis=-1)

start_h, start_w = 0, 0
end_h = H - h
end_w = W - w

n_h = H//s_x
n_w = W//s_y
#%%
for i in range(int(n_h-1)):
    for j in range(int(n_w-1)):
        new_start_h = start_h + i*s_y
        new_start_w = start_w + j*s_x
        Bands_smaller.append(BANDS[int(new_start_h):int(new_start_h+h),int(new_start_w):int(new_start_w+w),:])
        Bands_smaller_vid.append((BANDS_video[int(new_start_h):int(new_start_h+h),int(new_start_w):int(new_start_w+w),0:3]).astype(np.uint8))
        Bands_smaller_pca.append((BANDS_full[int(new_start_h):int(new_start_h+h),int(new_start_w):int(new_start_w+w),:]))
j = 0
#video = cv2.VideoWriter("../../vedbaek_movie.mp4", 0, 1, (w,h))

for i, l in zip(Bands_smaller,Bands_smaller_pca):
    #fig = plt.figure(figsize=(12.8,9.6),dpi=10)
    #ax = plt.gca()
    #ax.set_axis_off()
    #ax.imshow(i[:,:,0:3]/i[:,:,0:3].max())
    #plt.imsave(fname="../../vaerloese_sliced_680/"+str(j)+".png",arr = i[:,:,0:3],format='png') # Image for annotation
    np.save("../../vedbaek_sliced_680/"+str(j)+".npy",l) # All Bands

    j += 1
    #if j == 17:
    #    break

#plt.figure()
#plt.imshow(bands['RE'])
#plt.show()
