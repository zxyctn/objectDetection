import cv2 as cv
import numpy as np
from math import sqrt, ceil
import torch


def l2Normalize(features):
    # Normalization with Euclidean Metrics
    squared = np.square(features)
    sum_of_elements = np.sum(squared)
    norm = np.sqrt(sum_of_elements)

    # ||x|| + 0.0001 as the equation tells us
    # Firstly, we take the square root of the sum to get features vector's Euclidean length
    norm += 0.0001

    # Dividing all elements of the Feature Vector to get the Normalized vector
    l2Normalized = np.true_divide(features, norm)

    return l2Normalized


def padAndResize(img):
    height, width, channels = img.shape
    limit = ceil((max(height, width) - min(height, width)) / 2)

    limitHeight = limitWidth = 0

    if (height > width):
        limitWidth = limit
    else:
        limitHeight = limit

    padded = np.zeros((max(height, width), max(height, width), 3), np.float32)

    for i in range(height):
        for j in range(width):
            padded[i + limitHeight][j + limitWidth] = img[i][j]

    resized = cv.resize(padded, (224, 224))

    return resized


def normalizeImg(img):
    normalized = cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255

    for pixel in normalized:
        pixel -= np.array([0.485, 0.456, 0.406])
        pixel /= np.array([0.229, 0.224, 0.225])

    return normalized


def featureEx(img, model):
    # we append an augmented dimension to indicate batch_size, which is one
    img = np.reshape(img, [1, 224, 224, 3])

    # model takes as input images of size [batch_size, 3, im_height, im_width]
    img = np.transpose(img, [0, 3, 1, 2])

    # convert the Numpy image to torch.FloatTensor
    img = torch.from_numpy(img)

    # extract features
    feature_vector = model(img)

    return feature_vector
