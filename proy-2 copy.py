#%%
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# %%
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convEncoding(input):
    result = []
    for code in input:
        inner = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        inner[code] = 1.0
        result.append(inner)

    return result

def normalize(x):
    maximum = np.max(x)
    minimum = np.min(x)
    return(x - minimum) / (maximum - minimum)

# This is the function that returns data
# The load_data() function returns the label that is NOT one-hot encoding
def load_data(norm = False):
    files = [data_dir + str(i) for i in range(1, 6)]
    files_test = data_dir_test + 'test_batch'
    data_batch = unpickle(files[0])
    data_batch_test = unpickle(files_test)

    # image_data: the numpy array of combined image data from 5 batch files
    image_data = data_batch[b'data']
    image_data_test = data_batch_test[b'data']

    # label_data: array of labels combined from 5 batch files
    label_data = data_batch[b'labels']
    label_data_test = data_batch_test[b'labels']

    for file in files[1:5]:
        data_batch = unpickle(file)
        image_data = np.concatenate((image_data, data_batch[b'data']), axis=0)
        label_data += data_batch[b'labels']

    images_train = image_data
    images_test = image_data_test

    labels_train = np.asarray(label_data)
    labels_test = np.asarray(label_data_test)
    
    # read in names of classes
    class_names = ['airplane','automobile','bird','cat',
                   'deer','dog','frog','horse','ship','truck']

    if (norm == True):
        images_train = normalize(images_train)
        images_test = normalize(images_test)

    return images_train, images_test, labels_train, labels_test, class_names

# %%
data_dir = "cifar-10-batches-py/data_batch_"
data_dir_test = "cifar-10-batches-py/"
# labels are not one-hot-encoding!
images_train, images_test, labels_train, labels_test, class_names = load_data()
print("Data loaded: ")
print("==>training data shape:", images_train.shape)
print("==>test data shape:", images_test.shape)


# %%
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

# %%
# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)