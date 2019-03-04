import numpy as np
import sys, os
import time
import csv
from glob import glob
from tqdm import tqdm

sys.path.append(os.getcwd())


# crnn packages
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
# import models.crnn_resnet as crnn
import alphabets
str1 = alphabets.alphabet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='/home/mcc/data/Datafountain/text/verify_image')
parser.add_argument('--label_dir', type=str, default='/home/mcc/data/Datafountain/text/verify_label')
opt = parser.parse_args()


# crnn params
crnn_model_path = 'expr/80.pth'
alphabet = str1
nclass = len(alphabet)+1

def crnn_recognition(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')

    # h = int(image.size[1] / image.size[0] * (32 * 1.0))
    transformer = dataset.resizeNormalize((32, 320))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

def eval(results, labels):
    cnt = 0
    for pred in results:
        name = pred[0]
        if labels[name] == pred[1]:
            cnt += 1
    return cnt / len(labels)

if __name__ == '__main__':
    # labels
    labels = {}
    txt_list = glob(opt.label_dir + '/*.txt')
    for txt_path in txt_list:
        name = os.path.split(txt_path)[-1]
        with open(txt_path, 'r') as f:
            word = f.readline().strip()
            labels[name[:-4]] = word
    
    # predict
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    
    results = []
    image_list = glob(opt.image_dir + '/*.jpg')
    for image_path in tqdm(image_list):
        name = os.path.split(image_path)[-1]
        image = Image.open(image_path)
        pred = crnn_recognition(image, model)
        # print(pred)
        results.append([name[:-4], pred])

    print('accuracy : %.4f' % eval(results, labels))