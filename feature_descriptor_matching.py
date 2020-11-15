#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:30:56 2020

@author: mubariz
"""

import cv2
import numpy as np
from VPR_Techniques.HOG_VPR import compute_HOG_descriptor
from VPR_Techniques.AMOSNet import compute_AMOSNet_descriptor
#from VPR_Techniques.CALC import compute_CALC_descriptor
#from VPR_Techniques.AlexNet_VPR.AlexNet_VPR import compute_AlexNet_descriptor

total_query_images=1
total_ref_images=2

dataset_dir='sample_dataset/'

def read_images():
    ''' This function reads images from the dataset directory and returns them as lists of images. '''

    query_images=[]
    ref_images=[]
    
    for i in range(total_query_images):
        temp=cv2.imread(dataset_dir+'query/'+str(i)+'.jpg')  #Reads query image 0 from the database
        query_images.append(temp)
        
    for i in range(total_ref_images):
        temp=cv2.imread(dataset_dir+'ref/'+str(i*5)+'.jpg')  #Reads ref image 0 and ref image 5 from the database
        ref_images.append(temp)
        
    return query_images, ref_images

def compute_descriptors(query_images,ref_images):
    ''' This function takes query and reference images as the input and returns 4 different types of feature descriptors/encoders for each set. '''
    
    query_HOG_descriptors=[] #Histogram of Oriented Gradients
    query_AMOSNet_descriptors=[] #AMOSNet from the paper 'Deep Learning at Scale for VPR'
#    query_CALC_descriptors=[] #CALC from the paper 'Convolutional Autoencoder for Loop Closure'
#    query_AlexNet_descriptors=[] #The popular AlexNet with Gaussian Random Projection for descriptor formation from CNN Layers' activations
    
    ref_HOG_descriptors=[] #Histogram of Oriented Gradients
    ref_AMOSNet_descriptors=[] #AMOSNet from the paper 'Deep Learning at Scale for VPR'
#    ref_CALC_descriptors=[] #CALC from the paper 'Convolutional Autoencoder for Loop Closure'
#    ref_AlexNet_descriptors=[] #The popular AlexNet with Gaussian Random Projection for descriptor formation from CNN Layers' activations
    
    for query_image in query_images:
        query_HOG_descriptors.append(compute_HOG_descriptor(query_image))
        query_AMOSNet_descriptors.append(compute_AMOSNet_descriptor(query_image))
#        query_CALC_descriptors.append(compute_CALC_descriptor(query_image))
#        query_AlexNet_descriptors.append(compute_AlexNet_descriptor(query_image))
    
    for ref_image in ref_images:
        ref_HOG_descriptors.append(compute_HOG_descriptor(ref_image))
        ref_AMOSNet_descriptors.append(compute_AMOSNet_descriptor(ref_image))
#        ref_CALC_descriptors.append(compute_CALC_descriptor(ref_image))
#        ref_AlexNet_descriptors.append(compute_AlexNet_descriptor(ref_image))
        
    return  query_HOG_descriptors, query_AMOSNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors#, query_CALC_descriptors, query_AlexNet_descriptors, ref_CALC_descriptors, ref_AlexNet_descriptors

def feature_descriptor_matching(query_HOG_descriptors, query_AMOSNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors):   
    ''' Matching score between query image HOG descriptors and ref images HOG descriptors'''
    
    score=np.dot(query_HOG_descriptors[0].T,ref_HOG_descriptors[0])/(np.linalg.norm(query_HOG_descriptors[0])*np.linalg.norm(ref_HOG_descriptors[0]))
    print('HOG Query Image 0 and HOG Ref Image 0 Cosine Similarity:' + str(score))
    
    score=np.dot(query_HOG_descriptors[0].T,ref_HOG_descriptors[1])/(np.linalg.norm(query_HOG_descriptors[0])*np.linalg.norm(ref_HOG_descriptors[1]))
    print('HOG Query Image 0 and HOG Ref Image 5 Cosine Similarity:' + str(score))
    
    score=1-(np.sum(abs(np.subtract(query_AMOSNet_descriptors[0],ref_AMOSNet_descriptors[0])))/(256*256))  #Actually 1-L1 similarity
    print('AMOSNet Query Image 0 and AMOSNet Ref Image 0 L1 Similarity:' + str(score))
    
    score=1-(np.sum(abs(np.subtract(query_AMOSNet_descriptors[0],ref_AMOSNet_descriptors[1])))/(256*256))  #Actually 1-L1 similarity
    print('AMOSNet Query Image 0 and AMOSNet Ref Image 5 L1 Similarity:' + str(score))
    
def main():
    query_images,ref_images = read_images()
    query_HOG_descriptors, query_AMOSNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors = compute_descriptors(query_images,ref_images)
    print('All features computed!')
    feature_descriptor_matching(query_HOG_descriptors, query_AMOSNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors)
    
    '''  
    Tasks for you.
    1. Are the matching scores a correct reflection of what was expected? That is, are the matching scores between the correct matches higher than incorrect matches?
    2. Try to  implement a different matching function in feature_descriptor_matching, e.g. L2-norm and others that you can.
    3. What happens if you change the transpose (.T) in np.dot function from the query_HOG_descriptors to ref_HOG_descriptors. Make sure you understand the reasons behind the effects of this change. 
    '''
    
if __name__ == "__main__":
    #This condition is called when you execute 'python feature_descriptor_matching.py' from Linux terminal 
    main() 