from keras.models import Sequential
from keras.layers import Dense, Input, Lambda
from keras.layers.convolutional import Conv2D
import numpy as np



def diabetes_model():
    # fix random seed for reproducibility
    np.random.seed(7)

    # load pima indians dataset
    dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")

    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
