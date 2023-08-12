import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import ember
import lightgbm as lgb
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
    return  malconvsingle2.malconvmain(filename)
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
    test_generator = malconvsingle2.generator(newimg,labels_test,batch_size=10,shuffle=False)
    # test_generator = generator(hashes_test,batch_size=1,shuffle=False)
    test_p = malconvsingle2.mypredict_generator( test_generator)
    return test_p
    # return  malconvsingle2.malconvmain(filename)
    # retrun test1


def myemberpredict(filename):
    # prog = "classify_binaries"
    # descr = "Use a trained ember model to make predictions on PE files"
    # parser = argparse.ArgumentParser(prog=prog, description=descr)
    # parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    # parser.add_argument("-m", "--modelpath", type=str, default=None, required=True, help="Ember model")
    # parser.add_argument("binaries", metavar="BINARIES", type=str, nargs="+", help="PE files to classify")
    # args = parser.parse_args()
    # modelname="/home/gan/Downloads/mailwaredata/ember/ember2018/ember_model_2018.txt"
    modelname="/home/gan/Downloads/mailwaredata/ember/ember_2017_2/model.txt"
    # lgbm_model = lgb.Booster(model_file="/home/gan/Downloads/mailwaredata/ember/ember2018/ember_model_2018.txt")
    # modelname = "/home/gan/Downloads/mailwaredata/ember/ember2018/model.txt"
    if not os.path.exists(modelname):
        print("ember model {} does not exist".format(modelname))
    lgbm_model = lgb.Booster(model_file=modelname)
    newimg= np.zeros((filename.shape[0],filename.shape[1],1))
    # putty_data = open('/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/data/10', "rb").read()

    for i in range(filename.shape[0]):
        newimg[i,0:filename.shape[1],:]=filename[i,0:filename.shape[1],:,0]
    # # newimg= np.zeros((img.shape[0],2000000,1))
    # if img.shape[1] <  2000000:
    #     for i in range(img.shape[0]):
    #         newimg[i,0:img.shape[1],:]=img[i,0:img.shape[1],:,0]

    newimg=np.squeeze(newimg,axis=2).astype(np.uint8)
    #myinput = torch.from_numpy(newimg).cuda()
    # myinput = torch.LongTensor(newimg).cuda()
    labels_test=np.zeros((newimg.shape[0],1))
    # labels_test[0] = ember.predict_sample(lgbm_model, newimg[0].tobytes())
    for i in range(newimg.shape[0]):
        labels_test[i] = ember.predict_sample(lgbm_model, newimg[i].tobytes())
        print(labels_test[i])

    # putty_data = open(modelname, "rb").read()
    return labels_test

img = get_image('/home/gan/Desktop/12.exe')
img=np.expand_dims(img,axis=0)
img= img.T

def mysuperpix(img):
    #img = get_image(path)

    #k = len(img[0])
    k = img.shape[0]
    j=[]
    for i in range(k):
        j.append(i//4096)
#    return np.array(j)
    return np.expand_dims(j,axis=1)
    
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
# explanation = explainer.explain_instance(np.array((img)), 
#                                          myclassifycontent, # classification function
#                                          top_labels=2, 
#                                          hide_color=0, 
#                                          num_samples=1000,
#                                          segmentation_fn=mysuperpix) # number of images that will be sent to classification function


explanation = explainer.explain_instance(np.array((img)), 
                                         myemberpredict, # classification function
                                         top_labels=2, 
                                         hide_color=0, 
                                         num_samples=1000,
                                         segmentation_fn=mysuperpix) # number of images that will be sent to classification function

from skimage.segmentation import mark_boundaries
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
# img_boundry1 = mark_boundaries(temp/255.0, mask)
# plt.imshow(img_boundry1)
# plt.show()
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False)
# 
print("fuckyou")
print(explanation.local_exp)

#img_boundry2 = mark_boundaries(temp/255.0, mask)
#plt.imshow(img_boundry2)
#plt.show()