import argparse
import sys
import datetime
from importlib.metadata import entry_points

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt


NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
RUN = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_DIR = '/tmp/logs/fit' + RUN
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plugin_eps = entry_points(group="image_recognition.plugins")


def predict(path_to_model, x_test, y_test):
    model = tf.keras.saving.load_model(path_to_model)
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


try:
    model_factory = plugin_eps['model'].load()
except KeyError:
    def model_factory(input_shape, nb_classes): 
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(
                1, 1), activation='tanh', input_shape=input_shape, padding="same"),
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(16, kernel_size=(5, 5),
                                            strides=(1, 1), activation='tanh', padding='valid'),
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='tanh'),
            tf.keras.layers.Dense(84, activation='tanh'),
            tf.keras.layers.Dense(nb_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
   
        return model 


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
        return model

try:
    predict = plugin_eps['predict'].load()
except KeyError:
    pass


def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    return preprocess(
        x_train, y_train, x_test, y_test)


def main():
    if model_factory(INPUT_SHAPE, NUM_CLASSES) is None:
        print("No model factory!")
        sys.exit(1)

    parser = argparse.ArgumentParser('Machine Learning Trainer')

    parser.add_argument(
        '-e', '--epochs', help="Number of epochs to train for", dest='e')
    parser.add_argument('-o', '--model-output',
                        help="Directory where models are saved", dest='o', default='/tmp/models/')

    argv = parser.parse_args()
    epochs = int(argv.e)
    path_to_model = argv.o

    (x_train, y_train), (x_test, y_test) = get_data()

    model = model_factory(x_train[0].shape, NUM_CLASSES)
    print(model.summary())

    if epochs:
        model = train(model, x_train, y_train, x_test, y_test, epochs)

    model_filename = path_to_model + RUN + '.keras'
    model.save(model_filename, save_format='keras')
    print(f'Model is saved here: {model_filename}')


if __name__ == "__main__":
    main()
