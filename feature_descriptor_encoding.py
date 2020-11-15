'''
Created on Tue Nov 10 16:32:34 2020

@author: mubariz

This file provides a simple interface to read some query and reference images from the directory
and encodes them using various feature descriptors.
'''

import cv2
from VPR_Techniques.HOG_VPR import compute_HOG_descriptor
from VPR_Techniques.CALC import compute_CALC_descriptor
from VPR_Techniques.AMOSNet import compute_AMOSNet_descriptor
from VPR_Techniques.AlexNet_VPR.AlexNet_VPR import compute_AlexNet_descriptor

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
        
    return  query_HOG_descriptors, query_AMOSNet_descriptors, query_CALC_descriptors, query_AlexNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors, ref_CALC_descriptors, ref_AlexNet_descriptors
    
def main():
    query_images,ref_images = read_images()
    query_HOG_descriptors, query_AMOSNet_descriptors, query_CALC_descriptors, query_AlexNet_descriptors, ref_HOG_descriptors, ref_AMOSNet_descriptors,\
    ref_CALC_descriptors, ref_AlexNet_descriptors = compute_descriptors(query_images,ref_images)
    
    print('All features computed!')
    
    '''  
    Tasks for you.
    1. Try to find out the dimensions and types of the query and ref descriptors for all 4 VPR techniques
    2. Modify this code to compute the descriptors for only the first query and reference image      
    '''
    
if __name__ == "__main__":
    #This condition is called when you execute 'python feature_descriptor_encoding.py' from Linux terminal 
    main() 