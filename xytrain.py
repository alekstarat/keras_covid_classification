import os
import cv2
import numpy as np
from keras import utils


def get_xtrain_xtest():

    x_train = []
    x_test = []


    for filename in sorted(os.listdir('Covid19-dataset/train')):

        filepath = 'Covid19-dataset/train/' + filename

        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (200, 200))

        x_train.append(im)

    x_train = np.reshape(x_train, (251, 40000))
    x_train = x_train.astype('float32') / 255.


    for filename in sorted(os.listdir('Covid19-dataset/test/')):

        filepath = 'Covid19-dataset/test/' + filename

        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (200, 200))

        x_test.append(im)
        
    x_test = np.reshape(x_test, (66, 40000))
    x_test = x_test.astype('float32') / 255.

    return x_train, x_test


def get_ytrain_ytest():

    # covid - 1
    # normal - 2
    # pneumonia - 3

    Y_train = []
    Y_test = []

    for i in sorted(os.listdir('Covid19-dataset/train')):  # creating y_train
    
        if i[0]=='c': Y_train.append(0)

        elif i[0]=='n': Y_train.append(1)

        elif i[0]=='p': Y_train.append(2)


    for i in sorted(os.listdir('Covid19-dataset/test')):  # creating y_test
    
        if i[0]=='c': Y_test.append(0)

        elif i[0]=='n': Y_test.append(1)

        elif i[0]=='p': Y_test.append(2)
    
    y_train = np.array(Y_train)
    y_test = np.array(Y_test)

    y_train = utils.to_categorical(y_train, 3)
    y_test = utils.to_categorical(y_test, 3)
    
    return y_train, y_test