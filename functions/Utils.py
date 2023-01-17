#%%
# Script containing Utility functions for Drone data taken with MICA rededge camera 
# Assumes image(s) have been collected to one otrthomosaic using agisoft or other software
#%%
import matplotlib as mpl
from osgeo import gdal
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.stats import multivariate_normal as mvnormal
import numpy as np, cv2
import cv2 as cv
import matplotlib.pyplot as plt
from createROI import ROI
import os
import random
from matplotlib.colors import ListedColormap
from sklearn.utils import shuffle
import time
from tqdm import tqdm
import seaborn as sns

def normData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def cropImage(Bands,crop_fn,MIN,MAX,savefolder='.',show=False):
    # Crops Image from Lat Long coordinates
    H = list(Bands.values())[0].shape[0]
    W = list(Bands.values())[0].shape[1]

    LONG = np.linspace(MIN[0],MAX[0],W)
    LAT = np.linspace(MAX[1],MIN[1],H)

    cropped = np.loadtxt(crop_fn)

    LONG_min = np.argmin(np.abs(LONG - cropped[0,0]))
    LAT_min = np.argmin(np.abs(LAT - cropped[0,1]))
    LONG_max = np.argmin(np.abs(LONG - cropped[1,0]))
    LAT_max = np.argmin(np.abs(LAT - cropped[1,1]))

    for i in Bands:
        Bands[i] = Bands[i][LAT_min:LAT_max,LONG_min:LONG_max]

    fig, ax = plt.subplots()

    image = np.stack((normData(Bands['R']),normData(Bands['G']),normData(Bands['B'])),axis=-1)

    ax.imshow(image)
    ax.axis('off')

    fig.savefig(savefolder+'/cropped.png',dpi=300,bbox_inches='tight')

    if show == True:
        fig.show()


    # Returns cropped bands and new min max lat long coordinates
    return Bands, [cropped[0,0],cropped[1,1]], [cropped[1,0],cropped[0,1]]

def prepareData(Bands,roiFolder,regionfile, MIN,MAX,savefolder='.',show=False):
    # Prepares data for SVM and naive-Bayes

    files = os.listdir(roiFolder)
    classes = [i for i in files if not i.endswith(".txt")]
    area = [i for i in files if i.endswith(".txt")]
    
    H = list(Bands.values())[0].shape[0]
    W = list(Bands.values())[0].shape[1]
    
    LONG = np.linspace(MIN[0],MAX[0],W)
    LAT = np.linspace(MAX[1],MIN[1],H)

    if regionfile == False:
        region = np.array([[MIN[0],MIN[1]],[MAX[0],MAX[1]]])
    else:
        region = np.loadtxt(regionfile)

    LONG_min = np.argmin(np.abs(LONG - region[0,0]))
    LAT_min = np.argmin(np.abs(LAT - region[0,1]))
    LONG_max = np.argmin(np.abs(LONG - region[1,0]))
    LAT_max = np.argmin(np.abs(LAT - region[1,1]))
    
    Training = {}
    Region = {}
    count = 0


    # Region data
    for i in Bands:
        if regionfile == False:
            Region[i] = Bands[i] 
        else:
            Region[i] =  ((Bands[i])[LAT_min:LAT_max,LONG_min:LONG_max])

    Im_plot = np.zeros((H,W))
    label = 0
    
    # Training data from masks.
    for i in classes:
        
        rois = os.listdir(roiFolder+'/'+i)
        count = 0
        
        for j in rois:
        
            R = np.loadtxt(roiFolder+'/'+i+'/'+j)
            R_class =  ROI(R,LONG,LAT,Bands)
            R_class.getMask()
            R_class.mask_int[R_class.mask_int>0]+=label
            Im_plot += R_class.mask_int
            
            if count == 0:    
                hist = pd.DataFrame(R_class.getHist())
            else:
                hist = pd.concat((hist,pd.DataFrame(R_class.getHist())), ignore_index= True)
            
            count +=1
        
        Training[i] = hist
        label += 1
    
    labels = [i for i in range(len(classes))]

    random.seed(12)
    colors = []

    for i in range(len(classes)):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    my_cmap = ListedColormap(colors, name="my_cmap")
    

    # Plot results

    fig, ax = plt.subplots()
    Im_plot[Im_plot==0] = np.nan

    keys = list(Region.keys())
    if len(keys) == 1:
        image = normData(Region[keys[0]])
    else:
        image = np.stack((normData(Region[keys[0]]),normData(Region[keys[1]]),normData(Region[keys[2]])),axis=-1)
        image_full = np.stack((normData(Bands[keys[0]]),normData(Bands[keys[1]]),normData(Bands[keys[2]])),axis=-1)

    ax.imshow(image)
    ax.axis('off')
    ax.set_title("Region of Interest")
    fig.savefig(savefolder+'/region.png',dpi=300,bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.imshow(image_full)
    ax.imshow(Im_plot)
    ax.axis('off')
    ax.set_title("Training mask")
    fig.savefig(savefolder+'/masks.png',dpi=300,bbox_inches='tight')

    if show == True:
        plt.show()

    return Region, Training, classes, labels


def openTif(filename):
    # Open and read bands from .tif file
    # dataset contains several metadata
    # Bands are the bands values for each channel.
    try:
        dataset = gdal.Open(filename)
        print("Succefully loaded .tif file",'\n')
        print("-----------------------------------------------------------------")

    except:
        print("Did not load .tif file")
        print("-----------------------------------------------------------------")
    
    
    bands = ['B','G','R','RE','NIR']
    
    metadata = dataset.GetMetadata()
    Bands = {}
    
    for i in range(5):
        Bands[bands[i]] = dataset.GetRasterBand(i+1).ReadAsArray()
    
    return dataset,Bands

def pcaDrone(data,n_comp = 3,flag = 0,PCA_prev = None):
    # Performs PCA on drone dataset
    # Assumes input is of the raw data output from openTif() function, ie. Bands
    # n_comp can at max only be 5
    # flag = 0 == fit and transform data output data, 1 == fit only output pca, 2 == Transform existing PCA output data from pca.
    pca = PCA(n_comp)

    H,W = data['R'].shape

    d = {}
    bands = ['R','G','B','RE','NIR']
    channels = len(data)
    print("Applying PCA..")
    start_time = time.time()
    for i in range(channels):
        d[bands[i]] = data[bands[i]].reshape(H*W)
    df = pd.DataFrame(data=d)

    if flag==0:    
        pca = PCA(n_comp)
        data_PCA = pca.fit_transform(df)
        data_PCA.reshape(H,W,n_comp), pca
        Data_PCA = {}
        pcas = ['pc1','pc2','pc3']
        
        for i in range(n_comp):
            Data_PCA[pcas[i]] = data_PCA[:,i].reshape(H,W)
        print("Finished PCA in {:2.3f} seconds".format(time.time() - start_time))

        return Data_PCA, pca
    
    if flag == 1:
        pca = PCA(n_comp)
        pca.fit(df)
        return pca
    
    if flag == 2:
        data_PCA = PCA_prev(df)
        
        data_PCA.reshape(H,W,n_comp), pca
        Data_PCA = {}
        pcas = ['pc1','pc2','pc3']
        
        for i in range(n_comp):
            Data_PCA[pcas[i]] = data_PCA[:,:,i]
        print("Finished PCA in {:2.3f} seconds".format(time.time() - start_time))
        print("-----------------------------------------------------------------")
        return Data_PCA

def getMuSigma(training,classes):
    
    print("Calculating second order moments...")
    MU = {}
    SIGMA = {}

    for i in classes:
        MU[i]=(training[i].mean())
        SIGMA[i]=(training[i].cov())
        #print(r"For class {} $\mu =${} and $\Sigma = ${}".format(i,MU[i],SIGMA[i]))
    print("-----------------------------------------------------------------")
    return MU, SIGMA


def bayesPredict(data,MU,SIGMA,priors,classes,savefolder,show=False):
    # Using a bayesian classifier to classify drone data
    # input: data, data to be classified
    # MU is an array of arrays. each array consist of the mean of classisfication for each channels
    # ex. Classes if 2 classes then MU = [[mu1_1,mu1_2,mu1_3,mu1_4,mu1_5],[mu2_1,mu2_2,mu2_3,mu2_4,mu2_5]]
    # SIGMA is an array 2D arrays consisting of the covariance matrices of the classes.
    # priors are an array of priors for each class
    
    channels = list(data.keys())
    H = list(data.values())[0].shape[0]
    W = list(data.values())[0].shape[1]
    Data = np.zeros((H,W,len(channels)))

    if type(data) == dict:
        count = 0
        for i in channels:
            Data[:,:,count] = data[i]
            count += 1
    else:
        assert (data.ndim == 3) and (data.shape[2] <= 5)

    P = []
    count = 0
    print("Calculating log posteriors")
    for i in classes:
       pcX = (mvnormal(MU[i], SIGMA[i]).logpdf(Data)) + np.log(priors[count])
       P.append(pcX)
       count += 1
    print("Max posterior pixel prediction")
    print("-----------------------------------------------------------------")
    Pred = np.argmax(np.array(P),axis=0)
        
    random.seed(12)
    colors = []

    bound = [int(i) for i in range(len(classes))]
    Classes = classes.copy()
            
    for i in bound:
        if i in Pred:
            continue
        else:
            Classes.remove(classes[i])
            bound.remove(i)
    
    for i in range(len(Classes)):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    my_cmap = ListedColormap(colors, name="my_cmap")
    norm = mpl.colors.BoundaryNorm(bound, my_cmap.N)

    fig, ax = plt.subplots() 
    
    ims = ax.imshow(Pred,cmap=my_cmap)

    cbar = plt.colorbar(ims,ticks=bound, extend='both')
    print(bound,'\n',classes)
    cbar.set_ticks(bound)
    cbar.set_ticklabels(Classes)
    cbar.set_label("Classes")
    ax.axis('off')
    plt.title("Naive-Bayes Prediction")
    if show == True:
        fig.show()

    fig.savefig(savefolder+"/predict.png",dpi=300,bbox_inches='tight')

    return Pred


def bandsToDataFrame(Bands):
    # Transforms 2D mulitchannel data to 1D multichannel pandas dataframe
    H = list(Bands.values())[0].shape[0]
    W = list(Bands.values())[0].shape[1]
    bands = Bands.copy()    
    for i in Bands:
        bands[i] = bands[i].reshape(H*W)
    
    df = pd.DataFrame(bands)

    return df

def svmClassify(Train,Region,classes,Labels,H,W,savefolder,sample_ratio = 0.2,kern='rbf',show = False):
    # Uses a Support Vector Machine to classify drone data
    # Supervised learning
    clf = SVC(kernel = kern)


    # Preparing dataframe for training
    for i in range(len(classes)):
        lab = classes[i].lower()
        if i== 0:    
            t = Train[classes[i]].copy()
            l = np.zeros(len(Train[classes[i]]))+Labels[i]
            t_smp, l_smp = shuffle(t,l,random_state=0)

            n = len(l_smp)

            if lab!="trash":
                    
                training = t_smp[:int(n*sample_ratio)]
                labels = l_smp[:int(n*sample_ratio)]
            else:
                training = t_smp
                labels = l_smp

        else:
            t = Train[classes[i]].copy()
            l = np.zeros(len(Train[classes[i]]))+Labels[i]
            t_smp, l_smp = shuffle(t,l,random_state=0)
            n = len(l_smp)
            if lab != 'trash': 
                labels = np.concatenate((labels,l_smp[:int(n*sample_ratio)]))
                training = pd.concat((training,t_smp[:int(n*sample_ratio)]))
                #test = pd.concat((test,t_smp[int(n*sample_ratio):]))
            else:
                labels = np.concatenate((labels,l_smp))
                training = pd.concat((training,t_smp))
    
    training=(training-training.mean())/training.std()
    
    print("Number of samples for training: {}".format(training.shape[0]),'\n')
    print("Number of features for training: {}".format(training.shape[1]),'\n') 

    start_time = time.time()
    print("Training...",'\n')
    clf.fit(training,labels)
    train_time = time.time() - start_time
    print("Training Finished in: {:3.2f} seconds".format(train_time))
    print("-----------------------------------------------------------------")
    region = (Region.copy() - Region.copy().mean()) / (Region.copy().std())
    n = len(region)

    dn = 350000

    ite = int(np.round(n/dn) + 1)

    start_time = time.time()
    print("Predicting...",'\n')
    start = int(0)
    for i in tqdm(range(ite)):
        if i == 0:

            r_pred = region[start:start+dn]
            Test_pred = clf.predict(r_pred)
            start += dn
        else:
            r_pred = region[start:start+dn]
            Test_pred = np.concatenate((Test_pred,clf.predict(r_pred)))
            start += dn
        if start >= n:
            break

    predict_time = time.time() - start_time
    print("Predicting finished in: {:3.2f} seconds".format(predict_time))

    bound = [int(i) for i in range(len(classes))]
    Classes = classes.copy()      
    for i in bound:
        if i in Test_pred:
            continue
        else:
            Classes.remove(classes[i])
            bound.remove(i)
    random.seed(12)
    colors = []
    bound = [int(i) for i in range(len(Classes))]
    for i in range(len(Classes)):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    my_cmap = ListedColormap(colors, name="my_cmap")
     
    fig, ax = plt.subplots()
    ims = ax.imshow(Test_pred.reshape(H,W),cmap=my_cmap)
    cbar = plt.colorbar(ims,ticks=bound, extend='both',ax=ax)
    cbar.set_ticks(bound)
    cbar.set_ticklabels(Classes)
    ax.axis('off')
    plt.title("Support Vector Machines Prediction")
    fig.savefig(savefolder+"/predict_SVM.png",dpi=300,bbox_inches='tight')

    return Test_pred.reshape(H,W), clf, train_time, predict_time

def findContourAndCenter(Im, Classes, labels, minArea, savefolder, show=False):
    # Trash class should always be last class
    # 
    ImClass = Im.copy()

    #Im = Im.astype(np.uint8).copy() 

    trash_idx = np.argwhere(np.char.lower(np.array(Classes)) == "trash")
    #print(id)

    id = labels[trash_idx[0][0]]
    ImClass[ImClass!=id] = 0
    ImClass[ImClass==id] = 1
    
    bound = [int(i) for i in range(len(Classes))]
    classes = Classes.copy()
            
    for i in bound:
        if i in Im:
            continue
        else:
            classes.remove(Classes[i])
            bound.remove(i)

    n = len(classes)   
    ImClass = ImClass.astype(np.uint8)
    #ImClass = cv.medianBlur(ImClass,5)
    
    contours, hierarchy = cv.findContours(ImClass,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    centers = []
    contours_true = []
    count = 1
    H,W = Im.shape
    Im_rect = np.zeros((H,W)).astype(np.uint8)
    
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 8,
        }
    fig, ax = plt.subplots()
    
    for cnt in contours:

        if cv.contourArea(cnt) < minArea:
            continue

        elif cv.contourArea(cnt) >= minArea:
            print("Trash detected: {}".format(count))
            count += 1
            contours_true.append(cnt)
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX,cY])
            x,y,w,h = cv.boundingRect(cnt)
            if w > 50 or h > 50:
                continue
            Im_rect = cv.rectangle(Im_rect,(x,y),(x+w,y+h),(n+1),2)
            ax.text(x,y,'Trash',font)
    Im_rect = Im_rect.astype(np.float32)
    Im_rect[Im_rect == 0] = np.nan
    #print(ImClass)
    #print(ImClass.shape)
    random.seed(12)
    colors = []


    for i in range(len(classes)):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    my_cmap = ListedColormap(colors, name="my_cmap")
    
    ax.imshow(Im, cmap = my_cmap)
    ax.imshow(Im_rect)
    ax.axis('off')
    #cbar = plt.colorbar(ticks=bound, extend='both')
    #cbar.ax.set_yticklabels(Classes)
    ax.set_title('Detected Trash')

    fig2, ax2 = plt.subplots()
    ax2.imshow(ImClass)

    #if show == True: 
    #fig.show()
    #fig2.show()
    if show == True:
        plt.show()

    fig.savefig(savefolder+"/predict_Trash.png",dpi=300,bbox='tight',bbox_inches='tight')
    #Im_cnt = cv.drawContours(ImClass,contours_true,-1,(2),2)

    return centers

def extractFeatures(Train,Classes,Labels,savedir):
    classes = Classes.copy()
    # visualize class features, to determine colineratity between features
    N =  []
    for i in range(len(classes)):
        
        N.append(len(Train[classes[i]]))

    n = np.min(N)

    for i in range(len(classes)):
        if i == 0:    
            sz = len(Train[classes[i]][:n])
#            print(sz)
            t = Train[classes[i]].copy()
            t = t.sample(frac=1)
            t = t[:n]
            l = np.array([classes[i] for j in range(sz)])
#            print(l[:10],'\n')

            training = t
            labels = l

        else:
            sz = len(Train[classes[i]][:n])
#            print(sz)
            t = Train[classes[i]].copy()
            t = t.sample(frac=1)
            t = t[:n]
            l = np.array([classes[i] for j in range(sz)])
#            print(l[:10],'\n')
            #t_smp, l_smp = shuffle(t,l,random_state=0)
    
            labels = np.concatenate((labels,l))
            training = pd.concat((training,t))
    training = training.reset_index()
#    print(len(labels))
    training['Class'] = labels
    training = training.drop(columns="index")
    #idx = np.arange(0,len(training),1)

    #training = training[~training.index.duplicated()]

    #print(training)

    g = sns.pairplot(training,hue="Class")

    fig = g.fig

    fig.savefig(savedir+'/featuremap.png')
    #map_ = {}

    #for i in Labels:
    #    map_[str(i)] = classes[i]
    #print(map_)
    #category = pd.cut(training.Class, labels-1,classes)
    #print(category)

    #training['Class'] = training['Class'].map(map_) 

    return training

#def idxToLatLong(center,args):
if __name__ == '__main__':
    
    file = '../../Dronedata_Vaerloese/Vaerloese_Orthomosaic_noalpha.tif'

    ds, Bands = openTif(file)

    #Bands_pca = pcaDrone(Bands)

    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5] 
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3] 
    MIN = [minx,miny]
    MAX = [maxx,maxy]

    crop_fn = "./datasets/Vaerloese/crop.txt"

    Bands, MIN, MAX = cropImage(Bands,crop_fn,MIN,MAX)

    #Bands, pca = pcaDrone(Bands)
    r_folder = r'C:\Users\magnu\OneDrive - Danmarks Tekniske Universitet\Master 2. sem\Synthesis\Synthesis function testing\functions\datasets\Vaerloese\Polygons'

    region_f = "./datasets/Vaerloese/Region3.txt"
    #region_f = False
    Region, train, classes, labels = prepareData(Bands,r_folder,region_f,MIN,MAX,show = True)
    
    df_bands = bandsToDataFrame(Region)
    MU, SIGMA = getMuSigma(train,classes)

    H = list(Region.values())[0].shape[0]
    W = list(Region.values())[0].shape[1]


    training = extractFeatures(train,classes,labels)
    priors = [0.01,0.9,0.0010,0.07,0.0001,0.001]
    
    training = extractFeatures(train,classes,labels+'./')
    
    #print(training)
    #SVM_pred,clf,tt,pt = svmClassify(train,df_bands,classes,labels,H,W,'.',0.5)
    Bayes_pred = bayesPredict(Region,MU,SIGMA,priors,classes,'./',show=True)
    
    center = findContourAndCenter(Bayes_pred,classes,labels,20,'./',show=True)    
    #sample_ratio = 0.2


    #Bands = np.load("ImageOfInterest.npy",allow_pickle=True)
    #Image = np.loadtxt('Classtest.txt')
    #classes = [0,1,2]
    #area = 20
    
    #Im, Cent = findContourAndCenter(Image,classes,area)
    
    #plt.figure()
    #plt.imshow(Im)
    #plt.scatter(np.array(Cent)[:,0],np.array(Cent)[:,1],s=5,color='red')
    #plt.show()




# %%
