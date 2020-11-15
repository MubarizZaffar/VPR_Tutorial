#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:49:42 2020

@author: mubariz
"""
import cv2
import numpy as np

def compute_HOG_descriptor(query):
        
    winSize = (512,512)
    blockSize = (32,32)
    blockStride = (16,16)
    cellSize = (16,16)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    query_desc=hog.compute(cv2.resize(query, winSize))
    
    print(query_desc)
    print(query_desc.shape)
    
    return query_desc

def perform_HOG_VPR(query_desc,ref_map_features): 

    confusion_vector=np.zeros(len(ref_map_features))
    itr=0
    for ref_desc in ref_map_features:
        score=np.dot(query_desc.T,ref_desc)/(np.linalg.norm(query_desc)*np.linalg.norm(ref_desc))
        confusion_vector[itr]=score
        itr=itr+1
        
    return np.amax(confusion_vector), np.argmax(confusion_vector)