from torchvision import models as models
from pre_processing import padAndResize, normalizeImg, featureEx

import torch
import torch.nn as nn
import os
from glob import glob
import cv2 as cv
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True, progress=True)
modules = list(model.children())[:-1]
model = nn.Sequential(*modules)

for p in model.parameters():
    p.requires_grad = False

model = model.eval()
model = model.to(device)

trainImgs = []
testImgs = []

labels_train = []
features_train = []

start_dir = '../data/train'
for dir, _, _ in os.walk(start_dir):
    trainImgs.extend(glob(os.path.join(dir, "*.JPEG")))

start_dir = '../data/test'
for dir, _, _ in os.walk(start_dir):
    testImgs.extend(glob(os.path.join(dir, "*.JPEG")))

class_names = {
    'n01615121': 'eagle',
    'n02099601': 'dog',
    'n02123159': 'cat',
    'n02129604': 'tiger',
    'n02317335': 'starfish',
    'n02391049': 'zebra',
    'n02410509': 'bison',
    'n02422699': 'antelope',
    'n02481823': 'chimpanzee',
    'n02504458': 'elephant'
}

for f in trainImgs:
    if (f.split('/')[1] != 'train'):
        break;

    img = normalizeImg(padAndResize(cv.imread(f)))
    feature = featureEx(img, model)
    label = class_names[f.split('/')[2]]

    features_train.append(feature[0])
    labels_train.append(label)

torch.save(features_train, '../lib/features_train.pt')

with open('../lib/labels_train.json', 'w') as outfile:
    json.dump(labels_train, outfile)

features_test = []
labels_test = []

for f in testImgs:
    if (f.split('/')[1] != 'test'):
        break;

    img = normalizeImg(padAndResize(cv.imread(f)))

    feature = featureEx(img, model)
    features_test.append(feature[0])

torch.save(features_test, '../lib/features_test.pt')

for f in testImgs:
    name = f.split('/')[3]
    i = int(name.split('.')[0])

    if (i < 10):
        labels_test.append('starfish')
    elif (i < 20):
        labels_test.append('elephant')
    elif (i < 30):
        labels_test.append('zebra')
    elif (i < 40):
        labels_test.append('dog')
    elif (i < 50):
        labels_test.append('bison')
    elif (i < 60):
        labels_test.append('chimpanzee')
    elif (i < 70):
        labels_test.append('antelope')
    elif (i < 80):
        labels_test.append('cat')
    elif (i < 90):
        labels_test.append('eagle')
    else:
        labels_test.append('tiger')

with open('../lib/labels_test.json', 'w') as outfile:
    json.dump(labels_test, outfile)