from os import listdir
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import fnmatch
import numpy as np
import cv2

'''
path = "./yale-train-test/yale7/train/"

def find_files(directory, pattern):
    return [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]

pattern = 'subject*'  # 替换为你要查找的特定字符

files = find_files(path,pattern)

img_list = []
#new_dim = (32, 32)
for file in files:
    pixels = plt.imread(path+file)
    #resized_pixels = cv2.resize(pixels, new_dim, interpolation=cv2.INTER_LINEAR)
    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    img_list.append(pixels)


X = np.stack(img_list)
#X = X.astype(np.float32)
'''


#find the first 2D PC
def find_first_pc(X,eps=1e-6,maxitr=100):
    N, m, n = X.shape

    # initialize v,t
    w0 = np.random.normal(0,1,size=(n,1))
    w0 = w0/np.linalg.norm(w0, ord=2)
    itr = 0
    X_flat = np.reshape(X, (-1, X.shape[-1]))

    while itr <= maxitr:

        # calculate pij(t)
        p = np.dot(X_flat,w0)
        p = -1 + (p>=0)*2
        #p = p.reshape(N,m)
        # get updated w
        w1 = np.dot(X_flat.T, p)
        w1 = w1 / np.linalg.norm(w1, ord=2)

        #newly added on Nov 8, 2023, check whether w1 is orthogonal to any columns
        w1x = np.dot(X_flat, w1)

        if np.linalg.norm(w1 - w0, ord=2) < eps and np.count_nonzero(w1x) == len(w1x):
            break
        elif np.linalg.norm(w1 - w0, ord=2) < eps and np.count_nonzero(w1x) != len(w1x):
            w_st = np.random.normal(0, 1, size=(n, 1)) / 10000
            w0 = w1 + w_st
            w0 = w0 / np.linalg.norm(w0, ord=2)
        else:
            w0 = w1.copy()
        itr += 1
        #print(itr)
    return w1

#find the first L-th 2D PC
def find_k_pc(X,eps=1e-6,maxitr=100,L=4):
    w_list = []
    l = 1
    N, m, n = X.shape

    while (l <= L):
        w = find_first_pc(X, eps=eps, maxitr=maxitr)
        w_list.append(w)
        #update step
        #for i in range(N):
        #    for j in range(m):
        #        X[i,j,] = X[i,j,] - np.dot(np.dot(w,w.T),X[i,j,])
        X = X.reshape(N*m,n)
        X = X - np.dot(X,np.dot(w,w.T))
        X = X.reshape(N,m,n)
        l+=1
    w_list = np.stack(w_list)
    return w_list

#add zero padding to a 2D image
def zero_padded(image,top = 2,left = 2):
    h, w = image.shape
    new_h = h + 2*top
    new_w = w + 2*left
    padded_image = np.zeros((new_h, new_w))
    padded_image[top:top + h, left:left + w] = image
    return padded_image

#extract k*k patches from an image with zero padding
def extract_patches(image, k):

    pad_image = zero_padded(image,top = int(k/2),left = int(k/2))

    # 获取图像的宽度和高度
    h, w = pad_image.shape

    # 初始化一个空的列表来存储补丁
    patches = []

    # 对于图像中的每一个像素，提取一个k x k的补丁
    for i in range(0, h - k + 1):
        for j in range(0, w - k + 1):
            # 提取补丁
            patch = pad_image[i:i + k, j:j + k]
            patches.append(patch)

    return patches

#given all the images, return the required number of 2dpca filters
def get_filters(X,k=5,L=4,eps=1e-6,maxitr=100,seed=101):

    N, m, n = X.shape
    patches_bar_x = []
    patches_bar_y = []

    for i in range(N):
        image = X[i]
        patches = np.stack(extract_patches(image, k))
        # mean subtraction
        patches_mean = np.mean(patches, axis=0)
        patches_bar = patches - np.reshape(patches_mean, (1, k, k))

        # take the patches and its transpose
        patches_bar_x.append(patches_bar)
        patches_bar_y.append(np.transpose(patches_bar, axes=(0, 2, 1)))

    patches_bar_x = np.concatenate(patches_bar_x, axis=0)
    patches_bar_y = np.concatenate(patches_bar_y, axis=0)

    np.random.seed(seed)
    filter_x = find_k_pc(patches_bar_x, L=L, eps = eps, maxitr=maxitr)
    filter_y = find_k_pc(patches_bar_y, L=L, eps = eps, maxitr=maxitr)
    Ws = []
    for i in range(L):
        Ws.append(np.dot(filter_x[i], filter_y[i].T))
    Ws = np.stack(Ws)
    return Ws

#Given all the images as input,return the l12dpcanet filters at both layers
def get_l12dpcanet_filters(X,k=5,L1=4,L2=4,eps=1e-6,maxitr=100,seed=101):

    '''
    :param X: input images, N*m*n
    :param k: filter size
    :param L1: channel size/number of filters for the first layer
    :param L2: channel size/number of filters for the second layer
    :return: [W1,W2] W1:first layer filters, second layer filters
    '''
    N, m, n = X.shape  # N:number of images  m,n:height,width
    # get the first layer filters
    W_layer1 = get_filters(X, k=k, L=L1, eps=eps, maxitr=maxitr,seed=seed)

    # the output after the first convolutional layer
    O1 = np.zeros((N, m, n, L1))
    for i in range(N):
        O1i = np.zeros((m, n, L1))
        for j in range(L1):
            img = cv2.filter2D(X[i,], -1, W_layer1[j], borderType=cv2.BORDER_CONSTANT)  # 32*32
            O1i[:, :, j] = img
        O1[i] = O1i

    # get the filters of the second convolutional layer
    O1_combine = np.transpose(O1, (0, 3, 1, 2))
    O1_combine = O1_combine.reshape(-1, m, n)
    W_layer2 = get_filters(O1_combine, k=k, L=L2, eps=eps, maxitr=maxitr,seed=seed)
    return [W_layer1,W_layer2]

#Given all the images and two-layer filters as input,return the l12dpcanet feature vector for each image
def get_l12dpcanet_features(X,W,L1=4,L2=4,B=8):
    '''

    :param X: input images, N*m*n
    :param W: list of 2 layer filters, output from get_l12dpcanet_filters
    :param k: filter size
    :param L1: channel size/number of filters for the first layer
    :param L2: channel size/number of filters for the first layer
    :param B: block size
    :return: final feature vector: N * (2**L2*B*L1)
    '''
    N, m, n = X.shape  # N:number of images  m,n:height,width
    W_layer1 = W[0]
    W_layer2 = W[1]

    # the output after the first convolutional layer
    O1 = np.zeros((N, m, n, L1))
    for i in range(N):
        O1i = np.zeros((m, n, L1))
        for j in range(L1):
            img = cv2.filter2D(X[i,], -1, W_layer1[j], borderType=cv2.BORDER_CONSTANT)  # 32*32
            O1i[:, :, j] = img
        O1[i] = O1i

    # get the filters of the second convolutional layer
    O1_combine = np.transpose(O1, (0, 3, 1, 2))
    O1_combine = O1_combine.reshape(-1, m, n)

    # the output after the second convolutional layer
    O2 = np.zeros((N, m, n, L1, L2))
    for i in range(N):
        # the ith image
        O2i = np.zeros((m, n, L1, L2))
        for j in range(L1):
            O2ij = np.zeros((m, n, L2))
            for k in range(L2):
                img = cv2.filter2D(O1[i, :, :, j], -1, W_layer2[k], borderType=cv2.BORDER_CONSTANT)

                O2ij[:, :, k] = img

            O2i[:, :, j, :] = O2ij
        O2[i] = O2i

    # the pooling stage
    # Heaviside step function
    H2 = O2.copy()
    H2[H2 >= 0] = 1
    H2[H2 < 0] = 0

    for i in range(L2):
        H2[:, :, :, :, i] = H2[:, :, :, :, i] * (2 ** i)
    T = np.sum(H2, axis=-1)

    # divide T into B blocks
    bsize = int(m / B)
    Tblock = np.zeros((N, bsize, n, B, L1))
    for i in range(N):
        for j in range(L1):
            Tij = T[i, :, :, j]
            Tij_block = [Tij[i:i + bsize, ] for i in range(0, m, bsize)]
            Tij_block = np.stack(Tij_block, axis=-1)
            Tblock[i, :, :, :, j] = Tij_block

    # get histogram of occurences for each block, each channel of each image
    X_final = np.zeros((N, 2 ** L2, B, L1))
    for i in range(N):
        for j in range(B):
            for k in range(L1):
                b_ijk = Tblock[i, :, :, j, k]
                b_ijk = b_ijk.flatten()
                hist_ijk = np.histogram(b_ijk, bins=list(range(2 ** L2 + 1)))[0]
                X_final[i, :, j, k] = hist_ijk

    X_final = X_final.reshape(N, -1)

    return X_final





#Given all the images as input,return the l12dpcanet feature vector for each image
def get_l12dpcanet_features_oldversion(X,k=5,L1=4,L2=4,B=8,eps=1e-6,maxitr=100):
    '''

    :param X: input images, N*m*n
    :param k: filter size
    :param L1: channel size/number of filters for the first layer
    :param L2: channel size/number of filters for the first layer
    :param B: block size
    :param eps: tolerance level for l12dpca algorithm
    :param maxitr:  maximum iterations for  l12dpca algorithm
    :return: final feature vector: N * (2**L2*B*L1)
    '''
    N, m, n = X.shape  #N:number of images  m,n:height,width
    #get the first layer filters
    W_layer1 = get_filters(X,k=k,L=L1,eps=eps,maxitr=maxitr)

    # the output after the first convolutional layer
    O1 = np.zeros((N, m, n, L1))
    for i in range(N):
        O1i = np.zeros((m, n, L1))
        for j in range(L1):
            img = cv2.filter2D(X[i,], -1, W_layer1[j], borderType=cv2.BORDER_CONSTANT)  # 32*32
            O1i[:, :, j] = img
        O1[i] = O1i

    # get the filters of the second convolutional layer
    O1_combine = np.transpose(O1,(0,3,1,2))
    O1_combine = O1_combine.reshape(-1,m,n)
    W_layer2 = get_filters(O1_combine,k=k,L=L2,eps=eps,maxitr=maxitr)

    # the output after the second convolutional layer
    O2 = np.zeros((N, m, n, L1, L2))
    for i in range(N):
        # the ith image
        O2i = np.zeros((m, n, L1, L2))
        for j in range(L1):
            O2ij = np.zeros((m, n, L2))
            for k in range(L2):
                img = cv2.filter2D(O1[i, :, :, j], -1, W_layer2[k], borderType=cv2.BORDER_CONSTANT)

                O2ij[:, :, k] = img

            O2i[:, :, j, :] = O2ij
        O2[i] = O2i

    # the pooling stage
    # Heaviside step function
    H2 = O2.copy()
    H2[H2 >= 0] = 1
    H2[H2 < 0] = 0

    for i in range(L2):
        H2[:, :, :, :, i] = H2[:, :, :, :, i] * (2 ** i)
    T = np.sum(H2, axis=-1)

    # divide T into B blocks
    bsize = int(m / B)
    Tblock = np.zeros((N, bsize, n, B, L1))
    for i in range(N):
        for j in range(L1):
            Tij = T[i, :, :, j]
            Tij_block = [Tij[i:i + bsize, ] for i in range(0, m, bsize)]
            Tij_block = np.stack(Tij_block, axis=-1)
            Tblock[i, :, :, :, j] = Tij_block

    # get histogram of occurences for each block, each channel of each image
    X_final = np.zeros((N, 2 ** L2, B, L1))
    for i in range(N):
        for j in range(B):
            for k in range(L1):
                b_ijk = Tblock[i, :, :, j, k]
                b_ijk = b_ijk.flatten()
                hist_ijk = np.histogram(b_ijk, bins=list(range(2 ** L2 + 1)))[0]
                X_final[i, :, j, k] = hist_ijk

    X_final = X_final.reshape(N, -1)

    return X_final





#W = get_l12dpcanet_filters(X)
#X_final = get_l12dpcanet_features(X,W)



#X_final_1 = get_l12dpcanet_features_oldversion(X)










