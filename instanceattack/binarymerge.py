import os
import magic
import secml_malware
from secml.array import CArray
from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel

def singlemerge(filefirst, filesecond,length):

    binfile = open(filesecond,'rb')
    size= os.path.getsize(filesecond)
    
    if(length >size):
        return -1
    with open(filefirst, 'ab+') as f:
        for i in range(length):
            data = binfile.read(1)
            # print(data)
            f.write(data)
    binfile.close()
        # with open(filesecond, "b") as f2:

def foldermerge(folder, filesecond):
    for i, f in enumerate(os.listdir(folder)):
        path = os.path.join(folder, f)
        with open(path, "rb") as file_handle:
            # code = file_handle.read()
            singlemerge(path, filesecond,512)



foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv3')
foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv3')
foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv3')
foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv3')
foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv3')

# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')
# foldermerge('/home/gan/Downloads/mailwaredata/data/begnign', '/home/gan/Downloads/sourcecode/lime-master/doc/notebooks/tempdata/positiv')




