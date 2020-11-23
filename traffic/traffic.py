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
    images = []
    labels = []

    # Find each element within the given directory:
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)

        # Search the subdirectory for images if the category itself is a directory:
        if os.path.isdir(category_path):

            files = os.listdir(category_path)
            for filename in files:

                # Source: https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
                image = cv2.imread(os.path.join(category_path, filename))

                if image is not None:
                    # Source: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
                    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

                    if resized_image is not None:
                        images.append(resized_image)
                        labels.append(int(category))

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a neural network
    model = tf.keras.models.Sequential()

    # Add convolutional and max-pooling layers:
    for i in range(4):
        if i == 0:
            # Source: https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer
            model.add(tf.keras.layers.InputLayer((IMG_WIDTH, IMG_HEIGHT, 3)))
        else:
            # Convolutional and Max-pooling Layers:
            model.add(tf.keras.layers.Conv2D(IMG_WIDTH * i, 3, activation="relu"))
            model.add(tf.keras.layers.MaxPooling2D())

    # Flatten Layer:
    model.add(tf.keras.layers.Flatten())
    
    # Hidden Layer and Dropout to Prevent Overfitting
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES * 2, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.4))

    # Softmax Activated Output Layer
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    main()
