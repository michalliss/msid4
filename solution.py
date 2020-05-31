import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
import matplotlib.pyplot as plt


def prepare_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt', 'trousers', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(train_images)

    print(train_images[0])

    res = datagen.flow(train_images)
    print(res[0])

    return (train_images, train_labels), (test_images, test_labels)


def model_run(train_images, train_labels, test_images, test_labels):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(train_images)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # model.fit(train_images, train_labels, batch_size=64, epochs=10)

    model.fit(datagen.flow(train_images, train_labels, batch_size=32),
              steps_per_epoch=len(train_images) / 32, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Test accuracy: ", test_acc)


def main():
    (train_images, train_labels), (test_images, test_labels) = prepare_data()
    model_run(train_images, train_labels, test_images, test_labels)


if __name__ == '__main__':
    main()
