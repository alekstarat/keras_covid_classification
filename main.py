from venv import create
from xytrain import *
from model_check import *
from model import *


def run():

    x_train, x_test = get_xtrain_xtest()
    y_train, y_test = get_ytrain_ytest()

    modelCovid = create_model()
    compile_model(modelCovid)
    train_model(modelCovid, x_train, y_train)
    load_train(modelCovid)
    model_check(modelCovid, x_test, y_test)

if __name__ == '__main__':
    run()
