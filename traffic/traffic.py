import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    print("data_dir: ", data_dir)
    images = []
    labels = []
    # Find each element within the given directory:
    for category in os.listdir(data_dir):
        # print("category: ", category)
        category_path = os.path.join(data_dir, category)

        # Search the subdirectory for images if the category itself is a directory:
        if os.path.isdir(category_path):
            # print("isdir: ", category_path)

            files = os.listdir(category_path)
            # print("files: ", files)
            for filename in files:
                image = cv2.imread(os.path.join(category_path, filename))
                resized_image_data = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                # print("image: ", image)
                images.append(resized_image_data)
                labels.append(int(category))

    # print("images: ", images)
    # print("labels: ", labels)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(model.summary())
    return model

    #  model = tf.keras.models.Sequential([
    #     # tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    #     # tf.keras.layers.Conv2D(filters=NUM_CATEGORIES, kernel_size=3, strides=(2, 2), activation='relu'),
    #     # tf.keras.layers.Flatten(),
    #     # Convolutional layer:
    #     # tf.keras.layers.Conv2D(NUM_CATEGORIES, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu', padding='same'),
    #     tf.keras.layers.Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu'),

    #     # tf.keras.layers.Conv2DTranspose(NUM_CATEGORIES, 3, strides=2, padding='same'),
    #     # # Max-pooling layer, using 2x2 pool size
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    #     # tf.keras.layers.Conv2DTranspose(NUM_CATEGORIES, 3, strides=2, padding='same'),
    #     # # Max-pooling layer, using 2x2 pool size
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     # # Flatten units
    #     tf.keras.layers.Flatten(),

    #     # # # Add a hidden layer with dropout

    #     # tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dropout(0.5),

    #     # Add an output layer with output units for all possible categories:
    #     # tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #     # tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #     tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    # ])


    # TODO: could have training, calibration, and validation with a 3-part training test split
    # TODO: training time should be a few minutes
    # TODO: the nextwork should be 30 x 30 x 3 neurons
    # TODO: Look at banknotes for an example of a sequential + convolution network -- is this what I need to use?
    # TODO: should use relu activation for hidden layer
    # TODO: possibly use sigmoid for final layer?
    # TODO: https://www.tensorflow.org/tutorials/images/classification

    # Create a convolutional neural network
    # model = tf.keras.Sequential()
    # model.add()
    # model.add()
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    # model = tf.keras.models.Sequential([
    #     # tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    #     # tf.keras.layers.Conv2D(filters=NUM_CATEGORIES, kernel_size=3, strides=(2, 2), activation='relu'),
    #     # tf.keras.layers.Flatten(),
    #     # Convolutional layer:
    #     # tf.keras.layers.Conv2D(NUM_CATEGORIES, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu', padding='same'),
    #     tf.keras.layers.Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu'),

    #     # tf.keras.layers.Conv2DTranspose(NUM_CATEGORIES, 3, strides=2, padding='same'),
    #     # # Max-pooling layer, using 2x2 pool size
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    #     # tf.keras.layers.Conv2DTranspose(NUM_CATEGORIES, 3, strides=2, padding='same'),
    #     # # Max-pooling layer, using 2x2 pool size
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     # # Flatten units
    #     tf.keras.layers.Flatten(),

    #     # # # Add a hidden layer with dropout

    #     # tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dropout(0.5),

    #     # Add an output layer with output units for all possible categories:
    #     # tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #     # tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #     tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    # ])
    # model = tf.keras.models.Sequential([

    #     # Convolutional layer. Learn 32 filters using a 3x3 kernel
    #     tf.keras.layers.Conv2D(
    #         IMG_WIDTH, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    #     ),

    #     # Max-pooling layer, using 2x2 pool size
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    #     # Flatten units
    #     tf.keras.layers.Flatten(),

    #     # Add a hidden layer with dropout
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dropout(0.5),

    #     # Add an output layer with output units for all 10 digits
    #     tf.keras.layers.Dense(3, activation="softmax")
    # ])

if __name__ == "__main__":
    main()
