# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:13:51 2022

@author: magnu
"""
import numpy as np
import cv2
class ROI():
    
    def __init__(self,roi,Long,Lat,Image):
        self.roi = roi
        self.Long = Long
        self.Lat = Lat
        self.ROI_idx = roi.copy()
        self.Image = Image
        #self.H = self.Image['R'].shape[0]
        self.H =list(self.Image.values())[0].shape[0]
        #self.W = self.Image['R'].shape[1]
        self.W =list(self.Image.values())[0].shape[1]
        self.mask = np.zeros((self.H,self.W))
        self.mask_int = np.zeros((self.H,self.W))
    def getMask (self):
        for i in range(len(self.roi)):
            lon = np.argmin(np.abs(self.Long - self.roi[i,0]))
            lat = np.argmin(np.abs(self.Lat - self.roi[i,1]))
            self.ROI_idx[i,0] = lon
            self.ROI_idx[i,1] = lat
    
        self.ROI_idx = self.ROI_idx.astype(np.int32)
        cv2.fillConvexPoly(self.mask,self.ROI_idx,1)
        self.mask = self.mask.astype(bool)
        self.mask_int = self.mask.astype(int)
        
    def getHist(self):
        d_hist = {}
        for i in self.Image:
            Hist = self.Image[i][self.mask]
            d_hist[i] = Hist
            
        return d_hist
    def getCoords(self):
        coords = []
        for i in range(len(self.roi)):
            
            lon = np.argmin(np.abs(self.Long - self.roi[i,0]))
            lat = np.argmin(np.abs(self.Lat - self.roi[i,1]))
            
            coords.append([lon,lat])
        return coords
            
        
        
        
        
        
        
        
        
        
        
        