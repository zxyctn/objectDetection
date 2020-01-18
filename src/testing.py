from torchvision import models as models
from torch.utils.data import Dataset, DataLoader
from pre_processing import padAndResize, normalizeImg, featureEx

from custom_dataset import CustomDataset
from selective_search import selectiveSearch

import numpy as np
import torch
import torch.nn as nn
import os
from glob import glob
import cv2 as cv
from math import ceil
import json
from PIL import Image

def testing(FFNmodel, test_dataloader):
    model_state_dict = torch.load('../lib/model.pt')
    FFNmodel.load_state_dict(model_state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    FFNmodel = FFNmodel.to(device)
    FFNmodel.eval()

    k = 0    
    maxx = 0
    best = -1
    index = -1
    pred = 0

    for test in test_dataloader:
        test_features = test['features'][0]
        test_features = test_features.view(-1, 2048)
        test_labels = test['label']

        test_labels = test_labels.to(device)
        test_features = test_features.to(device)

        outputs = FFNmodel(test_features)

        x, predicted = torch.max(outputs.data, 1)
        best, i = torch.max(x, 0)

        if (maxx < best):
            maxx = best
            index = int(i) + k * 10
            pred = predicted.numpy()[int(i)]
        
        k += 1

    torch.save(FFNmodel.state_dict(), '../lib/model.pt')
    return index, pred

def testLoader(features_test, labels_test):
    test_dataset = CustomDataset(train=False, features=features_test, labels=labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)

    return test_dataloader

def getCropped(img, boxes):
    images = []

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    for b in boxes:
        x, y, w, h = b
        img_cropped = img.crop((x, y, (x +w), (y + h)))
        im_np = np.asarray(img_cropped)
        images.append(im_np)
    
    return images

def testImages(model) :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    modelResnet = models.resnet50(pretrained=True, progress=True)
    modules = list(modelResnet.children())[:-1]
    modelResnet = nn.Sequential(*modules)

    for p in modelResnet.parameters():
        p.requires_grad = False

    modelResnet = modelResnet.eval()
    modelResnet = modelResnet.to(device)

    testImgs = []
    labelsPredicted = []
    boxes = []

    start_dir = '../data/test'
    for dir, _, _ in os.walk(start_dir):
        testImgs.extend(glob(os.path.join(dir, "*.JPEG"))) 
    
    for f in testImgs:
        print(f)
        img = cv.imread(f)

        boundingBoxes = selectiveSearch(img, 'q', 100)
        cropped = getCropped(img, boundingBoxes)

        name = f.split('/')[4]
        ind = int(name.split('.')[0])

        labels_test = []

        if (ind < 10):
            labels_test.append('starfish')
        elif (ind < 20):
            labels_test.append('elephant')
        elif (ind < 30):
            labels_test.append('zebra')
        elif (ind < 40):
            labels_test.append('dog')
        elif (ind < 50):
            labels_test.append('bison')
        elif (ind < 60):
            labels_test.append('chimpanzee')
        elif (ind < 70):
            labels_test.append('antelope')
        elif (ind < 80):
            labels_test.append('cat')
        elif (ind < 90):
            labels_test.append('eagle')
        else:
            labels_test.append('tiger')
        
        features_test = []

        for i in cropped:
            i = normalizeImg(padAndResize(i))
            i = cv.cvtColor(i, cv.COLOR_RGB2BGR)
            features = featureEx(i, modelResnet)
            features_test.append(features)

        for j in range(len(cropped) - 1):
            labels_test.append(labels_test[0])

        loader = testLoader(features_test, labels_test)
        index, label = testing(model, loader)    

        labelsPredicted.append(label)
        boxes.append(boundingBoxes[index])
    
    return testImgs, labelsPredicted, boxes