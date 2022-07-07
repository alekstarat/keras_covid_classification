#import matplotlib.pyplot as plt
#from PIL import Image
import random
import numpy as np
from xytrain import get_xtest, get_ytrain_ytest


def model_check(model, x_test, y_test):

    n = random.randrange(0, 66)

    #plt.imshow(Image.fromarray(x_test[n]).convert('RGBA'))
    #plt.show()

    x = x_test[n]
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)
    pred = np.argmax(prediction)


    if pred == 0:

        print('Model predicted: ', pred, '(Covid-19)')
        print('True answer: ', y_test[n], '(Covid-19)')

    elif pred == 1:

        print('Model predicted: ', pred, '(Normal)')
        print('True answer: ', y_test[n], '(Normal)')

    elif pred == 2:

        print('Model predicted: ', pred, '(Pneumonia)')
        print('True answer: ', y_test[n], '(Pneumonia)')