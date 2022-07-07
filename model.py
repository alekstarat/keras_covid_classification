from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from xytrain import get_xtrain, get_xtest, get_ytrain_ytest


def create_model():

    modelCovid = Sequential()

    modelCovid.add(Dense(5000, input_dim=40000, activation='relu'))
    modelCovid.add(Dense(1000, activation='relu'))
    modelCovid.add(Dense(500, activation='relu'))
    modelCovid.add(Dense(3, activation='softmax'))

    return modelCovid

def compile_model(model):

    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    print(model.summary())


def train_model(model, x_train, y_train):

    model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=1)


def save_train(model):

    model.save_weights('modelCovid.h5')


def load_train(model):

    model.load_weights('modelCovid.h5')
