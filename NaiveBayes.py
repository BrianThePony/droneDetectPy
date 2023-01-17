#%%
import argparse
from pathlib import Path
import sys
import os
from Utils import openTif, pcaDrone, bayesPredict, prepareData, getMuSigma, findContourAndCenter, cropImage
import numpy as np
from createROI import ROI 
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

datasets = "./datasets"

def parse_opts(known=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data' , type = str, default = '/Vaerloese')
    parser.add_argument('--polygons', type=str, default=str(ROOT)+'/Polygons',help="Folder with polygons")
    parser.add_argument('--out', type=str, default= './runs/Naive-Bayes', help = 'saves to /runs/out')
    parser.add_argument('--priors', type=bool, default= False, help = 'If True, input priors for classes')
    parser.add_argument('--name', type = str, default = 'exp', help = "name of new directory")
    parser.add_argument('--region', type=str, default = '',help = "Path to region file")
    parser.add_argument('--crop', type=bool, default = False, help = "Crop Image, to remove redundant data. Must have .txt called 'crop.txt' with left and right corner coords")
    parser.add_argument('--pca', type=int,  default = 0, help = "Number of PCA components. if 0 no PCA is made")
    parser.add_argument('--RGB', type = bool, default=False, help= "Only use RGB bands")
    return  parser.parse_known_args()[0] if known else parser.parse_args()

def run(opts):


    # Preparing file names and folders
    folders = os.listdir(opts['out'])
    exp_folders = [i for i in folders if i.startswith('exp')]
    n = len(exp_folders)
    if opts['name'] == 'exp': # different from default
        os.makedirs(opts['out']+'/'+opts['name']+str(n),exist_ok=True)
        savedir = opts['out']+'/'+opts['name']+str(n)
    else:
        os.makedirs(opts['out']+'/'+opts['name'],exist_ok= True)
        savedir = opts['out']+'/'+opts['name']
    Crop_fn = datasets + opts['data']+'/crop.txt'
    Poly_fn = datasets+opts['data']+opts['polygons']

    files = os.listdir(datasets+opts['data'])
    data_fn = [datasets + opts['data']+'/'+i for i in files if i.endswith('.tif')]
    
    if opts['region'] == '':
        Reg_fn = False
    else:
        Reg_fn = datasets + opts["data"]+opts["region"]
    
    #print(savedir,'\n',data_fn,'\n',Crop_fn,'\n',Poly_fn,'\n',Reg_fn)

    # Load data
    ds, Bands = openTif(data_fn[0])

    # Coordinates of image
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5] 
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3] 
    MIN = [minx,miny]
    MAX = [maxx,maxy]

    if opts['RGB'] == True:
        del Bands['RE']
        del Bands['NIR']

    if opts['crop'] == True:
        Bands, MIN, MAX = cropImage(Bands,Crop_fn,MIN,MAX)
    if opts['pca'] == 1:
        Bands, pca = pcaDrone(Bands,1)
    elif opts['pca'] == 3:
        Bands, pca = pcaDrone(Bands,3)
    # Prepare training data and ROI
    Region, train, classes, labels = prepareData(Bands,Poly_fn,Reg_fn,MIN,MAX,savedir,False)
    MU, SIGMA = getMuSigma(train,classes)



    if opts['priors'] == True:
        priors = []

        n = len(classes)
        for i in range(n):
            print("Prior for '{}'".format(classes[i]))
            elem = float(input())
            priors.append(elem)
    else:
        # Equal priors
        priors = [1/len(classes) for i in range(len(classes))]
    
    # Prediction and trash location
    Bayes_pred = bayesPredict(Region,MU,SIGMA,priors,classes,savedir,show=False)
    center = np.array(findContourAndCenter(Bayes_pred,classes,labels,15,savedir,show=False))
    
    H = list(Region.values())[0].shape[0]
    W = list(Region.values())[0].shape[1]    

    # index to Lat Long
    if opts['region'] != '':
            
        startend = np.loadtxt(Reg_fn)

        MIN = [startend[0,0],startend[1,1]]
        MAX = [startend[1,0],startend[0,1]]    
        
        LONG = np.linspace(MIN[0],MAX[0],W)
        LAT = np.linspace(MAX[1],MIN[1],H)

    else:
        H = list(Bands.values())[0].shape[0]
        W = list(Bands.values())[0].shape[1]
        LONG = np.linspace(MIN[0],MAX[0],W)
        LAT = np.linspace(MAX[1],MIN[1],H)

    long = LONG[center[:,0]]
    lat = LAT[center[:,1]]

    d = {'Latitude [degrees]':lat,'Longitude [degrees]':long}
    
    df = pd.DataFrame(d)
    
    #Saving data
    df.to_csv(savedir+'/trashlocation.csv',index=False)
    print("Data saved to: "+ savedir)
    
if __name__ == '__main__':
    opts = parse_opts(True)
    #print(vars(opts))

    run(vars(opts))

