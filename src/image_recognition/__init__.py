import argparse 
import sys
import datetime
from importlib.metadata import entry_points

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt


NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
LOG_DIR = '/tmp/logs/fit' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plugin_eps = entry_points(group="image_recognition.plugins")

try:
    model_factory = plugin_eps['model'].load()
except KeyError:
    class LeNet(tf.keras.models.Sequential):
        def __init__(self, input_shape, nb_classes):
            super().__init__()
            self.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(
                1, 1), activation='tanh', input_shape=input_shape, padding="same"))
            self.add(tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding='valid'))
            self.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5),
                                            strides=(1, 1), activation='tanh', padding='valid'))
            self.add(tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding='valid'))
            self.add(tf.keras.layers.Flatten())
            self.add(tf.keras.layers.Dense(120, activation='tanh'))
            self.add(tf.keras.layers.Dense(84, activation='tanh'))
            self.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))

            self.compile(optimizer='adam',
                         loss=tf.keras.losses.categorical_crossentropy,
                         metrics=['accuracy'])
    model_factory = LeNet


try:
    preprocess = plugin_eps['preprocess'].load()
except KeyError:
    def preprocess(x_train, y_train, x_test, y_test):
        y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

        x_train = x_train[:, :, :, np.newaxis]
        x_test = x_test[:, :, :, np.newaxis]

        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255

        return (x_train, y_train), (x_test, y_test)


try:
    train = plugin_eps['train'].load()
except KeyError:
    def train(model, x_train, y_train, x_test, y_test, epochs=20):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, histogram_freq=1)

        model.fit(x_train, y=y_train, epochs=epochs,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard_callback])

try:
    predict = plugin_eps['predict'].load()
except KeyError:
    def predict(model, x_test, y_test):
        prediction_values = model.predict(x_test)

        # set up the figure
        fig = plt.figure(figsize=(15, 7))
        fig.subplots_adjust(left=0, right=1, bottom=0,
                            top=1, hspace=0.05, wspace=0.05)

        # plot the images: each image is 28x28 pixels
        for i in range(50):
            ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(x_test[i, :].reshape((28, 28)),
                      cmap=plt.cm.gray_r, interpolation='nearest')

            predicted_class = np.argmax(prediction_values[i])
            observed_class = np.argmax(y_test[i])
            if predicted_class == observed_class:
                # label the image with the blue text
                ax.text(0, 7, CLASS_NAMES[predicted_class], color='blue')
            else:
                # label the image with the red text
                ax.text(0, 7, CLASS_NAMES[predicted_class], color='red')
        plt.show()


def main():
    if model_factory(INPUT_SHAPE, NUM_CLASSES) is None:
        print("No model factory!")
        sys.exit(1)

    parser = argparse.ArgumentParser('Machine Learning Trainer') 

    parser.add_argument('-e', '--epochs', help="Number of epochs to train for", dest='e')  

    argv = parser.parse_args() 
    epochs = int(argv.e)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    (x_train, y_train), (x_test, y_test) = preprocess(
        x_train, y_train, x_test, y_test)

    model = model_factory(x_train[0].shape, NUM_CLASSES)
    print(model.summary())

    input('Press any key to continue...')

    if epochs: 
        train(model, x_train, y_train, x_test, y_test, epochs)

    while 1:
        s = input('Key in (p) to show sample predictions or (q) to quit')

        if s == 'p':
            predict(model, x_test, y_test)
            return
        elif s == 'q':
            print("Quitting!")
            return


if __name__ == "__main__":
    main()
