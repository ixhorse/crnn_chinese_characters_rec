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
# import models.crnn as crnn
import models.crnn_resnet as crnn
import alphabets
str1 = alphabets.alphabet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='/home/mcc/data/Datafountain/text/test')
parser.add_argument('--output_dir', type=str, default='/home/mcc/data/Datafountain')
opt = parser.parse_args()


# crnn params
crnn_model_path = 'expr/crnn_Rec_done_54_5890.pth'
alphabet = str1
nclass = len(alphabet)+1

def crnn_recognition(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')

    ## 
    # h = int(image.size[1] / image.size[0] * (32 * 1.0))
    transformer = dataset.resizeNormalize((32, 400))
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

if __name__ == '__main__':

	# crnn network
    model = crnn.CRNN(32, 1, nclass, 512)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    
    started = time.time()
    ## read an image
    with open(os.path.join(opt.output_dir, 'text_results.csv'), 'w') as f:
        writer = csv.writer(f)

        image_list = glob(opt.image_dir + '/*.jpg')
        for image_path in tqdm(image_list):
            name = os.path.split(image_path)[-1]
            image = Image.open(image_path)
            pred = crnn_recognition(image, model)
            # print(pred)
            writer.writerow([name, pred])

    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
    