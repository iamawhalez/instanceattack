#!/usr/bin/python
'''defines the MalConv architecture.
Adapted from https://arxiv.org/pdf/1710.09435.pdf
Things different about our implementation and that of the original paper:
 * The paper uses batch_size = 256 and SGD(lr=0.01, momentum=0.9, decay=UNDISCLOSED, nesterov=True )
 * The paper didn't have a special EOF symbol
 * The paper allowed for up to 2MB malware sizes, we use 1.0MB because of memory on a Titan X
 '''
import sys
import os
from io import StringIO
import numpy as np
import csv


def bytez_to_numpy(bytez,maxlen):
    b = np.ones( (maxlen,), dtype=np.uint16 )*padding_char
    bytez = np.frombuffer( bytez[:maxlen], dtype=np.uint8 )
    b[:len(bytez)] = bytez
    return b

def getfile_service(sha256,url=None,maxlen=maxlen):
    binary_path = sha256+''
    r = open(binary_path, "rb").read()
    
    return bytez_to_numpy( r, maxlen )        
        
def myfolder(rootDir):
    fileName ="/home/gan/Downloads/sourcecode/ember-master/malconvfoldersucess.csv"

    for root,dirs,files in os.walk(rootDir):
        for dir in dirs:
            myfolder(os.path.join(rootDir,dir))
        for file in files:
            if(file.endswith(".exe")==True):
                print(file)
                # print(malconvmain(os.path.join(root,file))[0][0])
                # return file,malconvmain(os.path.join(root,file))[0][0]
                with open(fileName,"a") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([file,malconvmain(os.path.join(root,file))[0][0]])
                # else:
                # return file, 0
    # return 0

if __name__ == '__main__':
    myfolder('/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/data/begnign')