import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt


def create_model(data_format='channels_first'):
    model = Sequential()

    # Define architecture of the model
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), data_format=data_format))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=data_format))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    # Give it a training objective
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def main():
    np.random.seed(123)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    fig = plt.figure(figsize=(8, 8))
    for i in range(16):
        img = X_train[i]
        fig.add_subplot(4, 4, i + 1)
        plt.imshow(img)

    plt.show()

    # Transform data to channel first shape
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize it
    X_train /= 255
    X_test /= 255

    # Convert labels into one-hot representation
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    model = create_model()
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=3, verbose=1)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print score

if __name__ == '__main__':
    main()
