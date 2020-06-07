
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import (Input, Activation, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization, AveragePooling2D, ZeroPadding2D)
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def prepare_data(n_val=5000, n_trans=1):

    # Load data
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Shuffle
    indexes = np.arange(x_train.shape[0])
    indexes = np.random.permutation(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]

    # Separate val data
    x_val = x_train[:n_val]
    y_val = y_train[:n_val]
    x_train = x_train[n_val:]
    y_train = y_train[n_val:]

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
    x_val = np.reshape(x_val, (x_val.shape[0], 28, 28, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

    # Transform images
    datagen = ImageDataGenerator(rotation_range=8,
                                 width_shift_range=0.08,
                                 shear_range=0.3,
                                 height_shift_range=0.08,
                                 zoom_range=0.08,
                                 preprocessing_function=get_random_eraser(
                                     v_l=0, v_h=1, pixel_level=True)
                                 )
    datagen.fit(x_train)

    x_trans, y_trans = datagen.flow(
        x_train[:n_trans], y_train[:n_trans], batch_size=n_trans)[0]

    x_train = np.concatenate([x_train[:], x_trans], axis=0)
    y_train = np.concatenate([y_train[:], y_trans], axis=0)

    # Shuffle 2
    indexes = np.arange(x_train.shape[0])
    indexes = np.random.permutation(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_model(layers, start_filters, initializer, dropout):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    filters = start_filters

    for i in range(layers):
        model.add(Conv2D(filters, (3, 3), padding="same",
                         kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2D(filters, (3, 3), padding="same",
                         kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

        filters = filters * 2

    model.add(Flatten())
    model.add(Dense(filters * 4))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation("softmax"))

    adam = tf.keras.optimizers.Adam()
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    plot_model(model, show_shapes=True, to_file='model.png')

    return model


def show_image(img):
    plt.imshow(image.array_to_img(img), cmap='gray', interpolation='nearest')
    plt.show()


def main():
    n_epochs = 50
    batch_size = 256
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data(6000, 60000)

    model = get_model(3, 64, 'he_normal', 0.25)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_data=(x_val, y_val))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    model.save('model')
    model.save_weights("model.h5")

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))


def load_ready():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data(6000, 1)

    model = get_model(3, 64, 'he_normal', 0.25)
    model.load_weights('model.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


load_ready()
# main()
