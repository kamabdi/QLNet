
"""
Created on Sat Jan  7 14:53:57 2017

@author: kamila

Load data(all .npy) for layer layer in data_folder
    Input: layer,data_folder
    Output: data

Choose patch pick num_regions=2  3x3xdepth regions and reshape data
    Input: data - batchxdepthxwidthxheight
    Output: Reshaped array of selected patches - nxm
 
"""

import numpy as np
import os
from random import randint
import torch as t



def choose_patch(data):
    kernel = 1
    dimention = np.shape(data)
    n = dimention[0]
    xdim = dimention[2]
    ydim = dimention[3]
    if xdim < kernel:
        kernel = 2
    selected_data = []
    num_regions = 4
    for i in range(0,n):
        for j in range(0,num_regions):
            x = randint(0,xdim-kernel)         
            y = randint(0,ydim-kernel)
            selected_data.append(data[i,:,x:(x+kernel),y:(y+kernel)])
    dimention = np.shape(selected_data)
    n = dimention[0]
    m = dimention[1]*dimention[2]*dimention[3]
    selected_data = np.reshape(selected_data, (n,m))
    return selected_data
    
def choose_single_patch(data):
    dimention = np.shape(data)
    n = dimention[0]
    xdim = dimention[2]
    ydim = dimention[3]
    selected_data = []
    num_regions = 1
    for i in range(0,n):
        for j in range(0,num_regions):
            x = 0         
            y = 0
            selected_data.append(data[i,:,x:(x+1),y:(y+1)])
    dimention = np.shape(selected_data)
    n = dimention[0]
    m = dimention[1]*dimention[2]*dimention[3]
    selected_data = np.reshape(selected_data, (n,m))
    return selected_data


def read_data(data_folder): 
    selected_data = []

    for file in os.listdir(data_folder):
        if file.endswith("0.npy")  or file.endswith("2.npy")or file.endswith("3.npy")or file.endswith("4.npy") or file.endswith("5.npy"):
        #if file.endswith("0.npy"):

            #        if file.endswith("100.npy"):
            
            data_var = t.load(data_folder +file)
            #data_arr = data_var.data.numpy()
            selected_data.append(data_var)
            #print np.shape(selected_data)
    
    return selected_data
            
    
def load_data(data_folder):	    
    selected_data = read_data(data_folder)
    dimention = np.shape(selected_data)
	    	
    print 'Dimention of data' + str(dimention)

    if len(dimention)==5:
        n = dimention[0]*dimention[1]
        selected_data = np.reshape(selected_data, (n,dimention[2],dimention[3],dimention[4]))
        selected_data = choose_patch(selected_data)
    elif len(dimention)==3 :
        n = dimention[0]*dimention[1]
        selected_data = np.reshape(selected_data, (n,dimention[2], 1, 1))
        selected_data = choose_single_patch(selected_data)
    else:
        n = dimention[0]
        selected_data = np.reshape(selected_data, (n,dimention[1],dimention[2],dimention[3]))
        selected_data = choose_patch(selected_data)
    
    dimention = np.shape(selected_data)

   # print dimention

    return selected_data

