# encoding: utf-8

import os
import sys
import csv
from functools import reduce

label_path = '/home/mcc/data/Datafountain/traindataset/train_lable.csv'

with open(label_path, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

text = [row[9] for row in data[1:]]

def add(a, b):
    return a + b

text = list(set(reduce(add, text)))

with open('./alphabets.py', 'w') as f:
    f.write('alphabet = \"\"\"')
    for word in text:
        f.write(word)
    f.write('\"\"\"')