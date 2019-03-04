# encoding: utf-8

import os
import sys
import csv
from functools import reduce

train_label_path = '/home/mcc/data/Datafountain/traindataset/train_lable.csv'
val_label_path = '/home/mcc/data/Datafountain/traindataset/verify_lable.csv'

with open(train_label_path, 'r') as f:
    reader = csv.reader(f)
    train_data = list(reader)

text = [row[9] for row in train_data[1:]]

with open(val_label_path, 'r') as f:
    reader = csv.reader(f)
    val_data = list(reader)

text += [row[9] for row in val_data[1:]]

def add(a, b):
    return a + b

text = list(set(reduce(add, text)))

with open('./alphabets.py', 'w') as f:
    f.write('alphabet = \"\"\"')
    for word in text:
        f.write(word)
    f.write('\"\"\"')