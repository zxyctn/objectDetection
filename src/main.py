# from training import FFNModel
import testing
import FFNmodel as ffn

import torch
import torch.nn as nn
import cv2 as cv
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FFNmodel = ffn.FFNModel(100)

model_state_dict = torch.load('../lib/model.pt')
FFNmodel.load_state_dict(model_state_dict)

FFNmodel = FFNmodel.to(device)

images, labels, boxes = testing.testImages(FFNmodel)

class_categories = ['eagle', 'dog', 'cat', 'tiger', 'starfish',
            'zebra', 'bison', 'antelope', 'chimpanzee', 'elephant']

if not os.path.exists('../results'):
    os.makedirs('../results')

os.chdir('../results')
for i in range(len(images)):
    img = cv.imread(images[i])

    name = images[i].split('/')[4]
    ind = name.split('.')[0]

    x, y, w, h = boxes[i]
    imTest = img.copy()

    cv.rectangle(imTest, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)
    imTest = cv.putText(imTest, class_categories[labels[i]], (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imwrite(f'../results/result_{ind}.png', imTest)

cv.destroyAllWindows()