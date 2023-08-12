import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import time
# import collections
###################
import malconvcallee
################
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        #with Image.open(f) as img:
        #    return img
        img=[]
        while True:
            data = f.read(1)
            #img.append(data)
            img.append(int.from_bytes(data,byteorder='little',signed= False))

            if data == b"":
                break
        # img = [[row[i] for row in img] for i in range(len(img[0]))]

        return img
            #return img.convert('RGB') 
def myclassify2(img):
    malconv = torch.load('checkpoint/example_sd_123.model')
    # if len(img)>2000000:
    #     img=img[0:2000000]
    # while len(img) < 2000000:
    #     img.append(0)
    newimg= np.zeros((img.shape[0],2000000,1))
    if img.shape[1] <  2000000:
        for i in range(img.shape[0]):
            newimg[i,0:img.shape[1],:]=img[i,0:img.shape[1],:,0]
    else:
        for i in range(2000000):
            newimg[i,0:2000000,:]=img[i,0:2000000,:,0]

    #myinput = torch.LongTensor([img]).cuda() 
    newimg=np.squeeze(newimg,axis=2)
    #myinput = torch.from_numpy(newimg).cuda()
    myinput = torch.LongTensor(newimg).cuda()
    sucess = malconv(myinput)
    return sucess.detach().cpu().numpy()
    # test = sucess.detach().cpu().numpy()
    # return test
def myclassify(filename):
    return  malconvcallee.malconvmain(filename)
def myclassifycontent(img):
    newimg= np.zeros((img.shape[0],2000000,1))
    if img.shape[1] <  2000000:
        for i in range(img.shape[0]):
            newimg[i,0:img.shape[1],:]=img[i,0:img.shape[1],:,0]
    else:
        for i in range(2000000):
            newimg[i,0:2000000,:]=img[i,0:2000000,:,0]
    newimg=np.squeeze(newimg,axis=2)
    #myinput = torch.from_numpy(newimg).cuda()
    myinput = torch.LongTensor(newimg).cuda()
    labels_test=np.zeros(newimg.shape[0])
    test_generator = malconvcallee.generator(newimg,labels_test,batch_size=10,shuffle=False)
    # test_generator = generator(hashes_test,batch_size=1,shuffle=False)
    test_p = malconvcallee.mypredict_generator( test_generator)
    return test_p
    # return  malconvsingle2.malconvmain(filename)
    # retrun test1

start_time=time.time()
img = get_image('/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/data/rof2/000af6ddc6ec97e2824cb2ebd5617ef03f8e6a207dcdcd855e4eafb32e2033d2.exe')
img=np.expand_dims(img,axis=0)
img= img.T

def mysuperpix(img):
    #img = get_image(path)

    #k = len(img[0])
    k = img.shape[0]
    j=[]
    for i in range(k):
        j.append(i//2048)
#    return np.array(j)
    return np.expand_dims(j,axis=1)

def mysuperpix2(img):
    #img = get_image(path)

    #k = len(img[0])
    k = img.shape[0]
    j=[]
    for i in range(k):
        j.append(i//2048)
#    return np.array(j)
    print(max(j))
    t=max(j)
    k=0
    for i in enumerate(j):
        # j.append(i//4096)
        if (i[1]==33):
            k=k+1
            if(k>=912 and k<=1424):
                j[i[0]]=(t+1)

    # print(max(j))
    # t=max(j)
    # k=0
    # for i in enumerate(j):
    #     # j.append(i//4096)
    #     if (i[1]==t):
    #         k=k+1
    #         if(k>=256):
    #             j[i[0]]=(t+1)
    # print(max(j))       

    # t=max(j)
    # k=0
    # for i in enumerate(j):
    #     # j.append(i//4096)
    #     if (i[1]==t):
    #         k=k+1
    #         if(k>=512):
    #             j[i[0]]=(t+1)
    # print(max(j)) 

    # t=max(j)
    # k=0
    # for i in enumerate(j):
    #     # j.append(i//4096)
    #     if (i[1]==t):
    #         # print(i)
    #         k=k+1
    #         if(k>=256):
    #             j[i[0]]=(t+1)
    # print(max(j))  

    # for i in enumerate(j):
    #     # j.append(i//4096)
    #     if (i[1]==t):
    #         # print(i)
    #         k=k+1
    #         if(k>=256):
    #             j[i[0]]=(t+1)
    # print(max(j))   
       
    return np.expand_dims(j,axis=1)


def mysuperpix3(img):
    #img = get_image(path)

    #k = len(img[0])
    k = img.shape[0]
    j=[]
    for i in range(k):
        j.append(i//4096)
#    return np.array(j)
    print(max(j))
    t=max(j)
    k=0
    for i in enumerate(j):
        # j.append(i//4096) 120
        if (i[1]==1):
            k=k+1
            if(k<=2048):
                j[i[0]]=(t+1)

    t=max(j)
    k=0
    for i in enumerate(j):
        # j.append(i//4096) 121
        if (i[1]==t):
            # print(i)
            k=k+1
            if(k<=1024):
                j[i[0]]=(t+1)
    print(max(j)) 

    t=max(j)
    k=0
    for i in enumerate(j):
        # j.append(i//4096) 121
        if (i[1]==t):
            # print(i)
            k=k+1
            if(k<=512):
                j[i[0]]=(t+1)
    print(max(j))     
    # print(max(j))
    t=max(j)
    k=0
    for i in enumerate(j):
        # j.append(i//4096)
        if (i[1]==t):
            k=k+1
            if(k<=256):
                j[i[0]]=(t+1)
    print(max(j))       


    return np.expand_dims(j,axis=1)
    
def mysuperpix4(img):
    #img = get_image(path)

    #k = len(img[0])
    k = img.shape[0]
    j=[]
    for i in range(k):
        j.append(i//4096)
#    return np.array(j)
    print(max(j))
    t=max(j)
    k=0
    for i in enumerate(j):
        # j.append(i//4096) 120
        if (i[1]==78):
            k=k+1
            if(k>=3840 ):#or k<=3584):
                j[i[0]]=(t+1)

    t=max(j)
    k=0
    for i in enumerate(j):
        # j.append(i//4096) 120
        if (i[1]==1):
            k=k+1
            if(k<=3584):
                j[i[0]]=(t+1)

    # t=max(j)
    # k=0
    # for i in enumerate(j):
    #     # j.append(i//4096) 121
    #     if (i[1]==t):
    #         # print(i)
    #         k=k+1
    #         if(k<=1024):
    #             j[i[0]]=(t+1)
    # print(max(j)) 

    # t=max(j)
    # k=0
    # for i in enumerate(j):
    #     # j.append(i//4096) 121
    #     if (i[1]==t):
    #         # print(i)
    #         k=k+1
    #         if(k<=512):
    #             j[i[0]]=(t+1)
    # print(max(j))     
    # # print(max(j))
    # t=max(j)
    # k=0
    # for i in enumerate(j):
    #     # j.append(i//4096)
    #     if (i[1]==t):
    #         k=k+1
    #         if(k<=256):
    #             j[i[0]]=(t+1)
    # print(max(j))       


    return np.expand_dims(j,axis=1)
    

from lime import lime_image
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array((img)), 
                                         myclassifycontent, # classification function
                                         top_labels=2, 
                                         hide_color=0, 
                                         num_samples=1000,
                                         segmentation_fn=mysuperpix2) # number of images that will be sent to classification function

from skimage.segmentation import mark_boundaries
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
# img_boundry1 = mark_boundaries(temp/255.0, mask)
# plt.imshow(img_boundry1)
# plt.show()
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False)
# for i in range(17409):
#     if mask[i]==1:
#         print(i)
#print("fuckyou")
print(explanation.local_exp)#<class 'dict'>
end_time=time.time()
print(end_time-start_time)
#img_boundry2 = mark_boundaries(temp/255.0, mask)
#plt.imshow(img_boundry2)
#plt.show()