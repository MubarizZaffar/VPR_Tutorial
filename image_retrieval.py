#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:32:12 2020

@author: mubariz
"""

import cv2
import numpy as np
from VPR_Techniques.HOG_VPR import compute_HOG_descriptor, perform_HOG_VPR
from VPR_Techniques.AMOSNet import compute_AMOSNet_descriptor, perform_AMOSNet_VPR
from VPR_Techniques.CALC import compute_CALC_descriptor, perform_CALC_VPR
from VPR_Techniques.AlexNet_VPR.AlexNet_VPR import compute_AlexNet_descriptor, perform_AlexNet_VPR

total_query_images=10
total_ref_images=10

dataset_dir='sample_dataset/'

def read_images():
    ''' This function reads images from the dataset directory and returns them as lists of images. '''

    query_images=[]
    ref_images=[]
    
    for i in range(total_query_images):
        temp=cv2.imread(dataset_dir+'query/'+str(i)+'.jpg')  
        query_images.append(temp)
        
    for i in range(total_ref_images):
        temp=cv2.imread(dataset_dir+'ref/'+str(i)+'.jpg')  
        ref_images.append(temp)
        
    return query_images, ref_images

def compute_descriptors(query_images,ref_images):
    ''' This function takes query and reference images as the input and returns 4 different types of feature descriptors/encoders for each set. '''
    
    query_HOG_descriptors=[] #Histogram of Oriented Gradients
    query_AMOSNet_descriptors=[] #AMOSNet from the paper 'Deep Learning at Scale for VPR'
    query_CALC_descriptors=[] #CALC from the paper 'Convolutional Autoencoder for Loop Closure'
    query_AlexNet_descriptors=[] #The popular AlexNet with Gaussian Random Projection for descriptor formation from CNN Layers' activations
    
    ref_HOG_descriptors=[] #Histogram of Oriented Gradients
    ref_AMOSNet_descriptors=[] #AMOSNet from the paper 'Deep Learning at Scale for VPR'
    ref_CALC_descriptors=[] #CALC from the paper 'Convolutional Autoencoder for Loop Closure'
    ref_AlexNet_descriptors=[] #The popular AlexNet with Gaussian Random Projection for descriptor formation from CNN Layers' activations
    
    for query_image in query_images:
        query_HOG_descriptors.append(compute_HOG_descriptor(query_image))
        query_AMOSNet_descriptors.append(compute_AMOSNet_descriptor(query_image))
        query_CALC_descriptors.append(compute_CALC_descriptor(query_image))
        query_AlexNet_descriptors.append(compute_AlexNet_descriptor(query_image))
    
    for ref_image in ref_images:
        ref_HOG_descriptors.append(compute_HOG_descriptor(ref_image))
        ref_AMOSNet_descriptors.append(compute_AMOSNet_descriptor(ref_image))
        ref_CALC_descriptors.append(compute_CALC_descriptor(ref_image))
        ref_AlexNet_descriptors.append(compute_AlexNet_descriptor(ref_image))
        
    return  query_HOG_descriptors, query_AMOSNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors, \
            query_CALC_descriptors, query_AlexNet_descriptors, ref_CALC_descriptors, ref_AlexNet_descriptors


def image_retrieval(query_HOG_descriptors, query_AMOSNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors, \
                    query_CALC_descriptors, query_AlexNet_descriptors, ref_CALC_descriptors, ref_AlexNet_descriptors):
    itr=0
    print('Image Retrieval for HOG:-')
    for query_desc in query_HOG_descriptors:
        match_score, match_index = perform_HOG_VPR(query_desc, ref_HOG_descriptors)
        print ('Query Image: '+str(itr)+' matches reference image: '+str(match_index),'Matching Confidence: ', str(match_score))
        itr=itr+1

    itr=0
    print('Image Retrieval for AMOSNet:-')
    for query_desc in query_AMOSNet_descriptors:
        match_score, match_index = perform_AMOSNet_VPR(query_desc, ref_AMOSNet_descriptors)
        print ('Query Image: '+str(itr)+' matches reference image: '+str(match_index),'Matching Confidence: ', str(match_score))
        itr=itr+1

    itr=0
    print('Image Retrieval for CALC:-')
    for query_desc in query_CALC_descriptors:
        match_score, match_index = perform_CALC_VPR(query_desc, ref_CALC_descriptors)
        print ('Query Image: '+str(itr)+' matches reference image: '+str(match_index),'Matching Confidence: ', str(match_score))
        itr=itr+1

    itr=0
    print('Image Retrieval for AlexNet:-')
    for query_desc in query_AlexNet_descriptors:
        match_score, match_index = perform_AlexNet_VPR(query_desc, ref_AlexNet_descriptors)
        print ('Query Image: '+str(itr)+' matches reference image: '+str(match_index),'Matching Confidence: ', str(match_score))
        itr=itr+1

def main():
    query_images,ref_images = read_images()
    query_HOG_descriptors, query_AMOSNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors, query_CALC_descriptors, \
    query_AlexNet_descriptors, ref_CALC_descriptors, ref_AlexNet_descriptors = compute_descriptors(query_images,ref_images)
    print('All features computed!')    
    
    print('Performing Image Retrieval (VPR) now!')
    image_retrieval(query_HOG_descriptors, query_AMOSNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors, \
                    query_CALC_descriptors, query_AlexNet_descriptors, ref_CALC_descriptors, ref_AlexNet_descriptors)
    
    '''  
    Tasks for you.
    1. Instead of just printing the matching indices, create a collage of images for all VPR techniques where the first row is 10 query images and the subsequent rows are 10 best matching images retrieved by each VPR technique. See PyPlot documentation online, for example.
    2. Instead of just selecting the best matching candidate, modify this code such that you output a list of Top-K matching candidates for each VPR technique for each query image.
    3. Given the results retrieved in this code, try to compute the Precision and Recall values for each VPR technique. 
    '''

if __name__ == "__main__":
    #This condition is called when you execute 'python image_retrieval.py' from Linux terminal 
    main()
    