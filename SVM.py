#%%
import argparse
from pathlib import Path
import sys
import os
from functions.Utils import openTif, bandsToDataFrame, prepareData,pcaDrone,svmClassify, findContourAndCenter, cropImage
import numpy as np
from functions.createROI import ROI 
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

datasets = str(ROOT) + '/datasets'

def parse_opts(known=False):

    parser = argparse.ArgumentParser()
    #change to: str(ROOT) + '/datasets/Vaerloese/drone.tif'
    #parser.add_argument('--data', type=str, 
    #                    default='/Vaerloese_Orthomosaic_noalpha.tif',
    #                    help='.tif data filename') 
    parser.add_argument('--polygons',type=str,default='/Polygons'
                        ,help="Folder with polygons")
    parser.add_argument('--data' , type = str, default = '/Vaerloese')
    parser.add_argument('--out',type=str,default= './runs/SVM', help = 'saves to /runs/SVM')
    parser.add_argument('--name',type = str, default = 'exp', help = "name of output folder")
    parser.add_argument('--region',type=str, default='',help = "Path to region file")
    parser.add_argument('--pca',type=int,default = 0, help = "Number of PCA components. if 0 no PCA is made")
    parser.add_argument('--crop',type = bool, default=False, 
                        help = "Crop Image, to remove redundant data. Must have .txt called 'crop.txt' with left and right corner coords")
    parser.add_argument('--splitsize', type = float, default = 0.5, help="Set sample ration, can help speed things up")
    parser.add_argument('--kernel', type = str, default = 'rbf',
    help="Set the kernel for the SVM classifier. The following kernels are available: 'linear', 'poly', 'rbf', 'sigmoid'")
    parser.add_argument('--RGB', type = bool, default=False, help= "Only use RGB bands")


    return  parser.parse_known_args()[0] if known else parser.parse_args()
def run(opts):

    folders = os.listdir(opts['out'])
    exp_folders = [i for i in folders if i.startswith('exp')]
    n = len(exp_folders)
    if opts['name'] == 'exp': # different from default
        os.makedirs(opts['out']+'/'+opts['name']+str(n),exist_ok=True)
        savedir = opts['out']+'/'+opts['name']+str(n)
    else:
        os.makedirs(opts['out']+'/'+opts['name'],exist_ok=True)
        savedir = opts['out']+'/'+opts['name']
    Crop_fn = datasets + opts['data']+'/crop.txt'
    Poly_fn = datasets+opts['data']+opts['polygons']

    files = os.listdir(datasets+opts['data'])
    data_fn = [datasets + opts['data']+'/'+i for i in files if i.endswith('.tif')]
    
    if opts['region'] == '':
        Reg_fn = False
    else:
        Reg_fn = datasets + opts["data"]+opts["region"]
    
    print(savedir,'\n',data_fn,'\n',Crop_fn,'\n',Poly_fn,'\n',Reg_fn)

    ds, Bands = openTif(data_fn[0])

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
    
    Region, train, classes, labels = prepareData(Bands,Poly_fn,Reg_fn,MIN,MAX,savedir,False)
    df_bands = bandsToDataFrame(Region)
    
    H = list(Region.values())[0].shape[0]
    W = list(Region.values())[0].shape[1]

    SVM_pred, clf,tt,pt = svmClassify(train,df_bands,classes,labels,H,W,savedir,opts['splitsize'],opts['kernel'])
    
    center = np.array(findContourAndCenter(SVM_pred,classes,labels,15,savedir,show=False))
    
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

    df.to_csv(savedir+'/trashlocation.csv')
    
    print("Data saved to: "+ savedir)

if __name__ == '__main__':
    opts = parse_opts(True)
    print(vars(opts))

    run(vars(opts))
