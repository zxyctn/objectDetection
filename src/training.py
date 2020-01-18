from torchvision import models as models
from torch.utils.data import Dataset, DataLoader

from custom_dataset import CustomDataset
from FFNmodel import FFNModel

import numpy as np
import torch
import torch.nn as nn
import os
from glob import glob
import cv2 as cv
from math import ceil
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

start_dir = '../data/train'
for dir, _, _ in os.walk(start_dir):
    trainImgs.extend(glob(os.path.join(dir, "*.JPEG")))

start_dir = '../data/test'
for dir, _, _ in os.walk(start_dir):
    testImgs.extend(glob(os.path.join(dir, "*.JPEG")))

features_test = torch.load('../lib/features_test.pt')
features_train = torch.load('../lib/features_train.pt')

input_labels_train = open('../lib/labels_train.json')
labels_train = json.load(input_labels_train)

input_labels_test = open('../lib/labels_test.json')
labels_test = json.load(input_labels_test)

train_dataset = CustomDataset(
    train=True, features=features_train, labels=labels_train)
test_dataset = CustomDataset(
    train=False, features=features_test, labels=labels_test)

train_dataloader = DataLoader(
    train_dataset, batch_size=10, shuffle=True, num_workers=4)
test_dataloader = DataLoader(
    test_dataset, batch_size=10, shuffle=False, num_workers=4)

FFNmodel = FFNModel(100)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(FFNmodel.parameters(), lr=0.001, momentum=0.9)

batch_size = 100
n_iters = 3000
num_epochs = ceil(n_iters / (len(features_test) / batch_size))
num_epochs = int(num_epochs)

model_state_dict = torch.load('../lib/model.pt')
FFNmodel.load_state_dict(model_state_dict)

FFNmodel = FFNmodel.to(device)

iter = 0
for epoch in range(num_epochs):
    for i, sample in enumerate(train_dataloader):
        FFNmodel.train()

        labels = sample['label']
        features = sample['features'][0]
        
        features = features.view(-1, 2048)

        labels.to(device)
        features.to(device)

        optimizer.zero_grad()

        outputs = FFNmodel(features)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        iter += 1

        if iter % 100 == 0:
            FFNmodel.eval()

            correct = 0
            total = 0
            for test in test_dataloader:
                test_features = test['features'][0]
                test_features = test_features.view(-1, 2048)
                test_labels = test['label']

                outputs = FFNmodel(test_features)
                x, predicted = torch.max(outputs.data, 1)

                total += test_labels.size(0)

                correct += (predicted == test_labels).sum()

            accuracy = 100 * correct / total

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(
                iter, loss.item(), accuracy))

            torch.save(FFNmodel.state_dict(), '../lib/model.pt')
