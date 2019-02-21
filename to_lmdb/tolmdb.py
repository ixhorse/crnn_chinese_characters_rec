import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import re
from PIL import Image
import numpy as np
import imghdr
import glob
import tqdm
import pdb


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False		
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            try:
                txn.put(k, v)
            except:
                pdb.set_trace()
			
def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        # imagePath = ''.join(imagePathList[i]).split()[0].replace('\n','').replace('\r\n','')
        #print(imagePath)
        label = ''.join(labelList[i])
        print(label)
        # if not os.path.exists(imagePath):
        #     print('%s does not exist' % imagePath)
        #     continue	
		
        with open(imagePath, 'r') as f:
            imageBin = f.read()


        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        print(cnt)
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
	

if __name__ == '__main__':
    home_path = os.path.expanduser('~')
    data_root = os.path.join(home_path, 'data/Datafountain')
    text_path = os.path.join(data_root, 'text')
    text_image_dir = os.path.join(text_path, 'image')
    text_label_dir = os.path.join(text_path, 'label')
    outputPath = "./lmdb"

    imagePathList = glob.glob(text_image_dir + '/*.jpg')
    labelList = []
    print('reading label txt...')
    for img_path in tqdm.tqdm(imagePathList):
        name = os.path.split(img_path)[-1][:-4] + '.txt'
        label_path = os.path.join(text_label_dir, name)
        with open(label_path, 'r') as f:
            words = f.readline()
            labelList.append(words)
    print('done.')

    createDataset(outputPath, imagePathList, labelList)

