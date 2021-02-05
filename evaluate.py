"""Evaluation on the Labeled Faces in the Wild dataset

This file is modified from the ArcFace official implementation in order to run
with TensorFlow. You can find the original file here:
https://github.com/deepinsight/insightface/blob/master/recognition/ArcFace/verification.py
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from predict import WanderingAI


def load_bin(filepath, image_size=[112, 112]):
    """Load the images and labels from the binary file.

    Args:
        filepath: the path to the binary file.
        image_size: the size fo the image.

    Returns:
        images and labels.
    """
    print("Loading the test files..")
    with open(filepath, 'rb') as f:
        bins, labels = pickle.load(f, encoding='bytes')

    images = []
    for bin in bins:
        images.append(tf.image.decode_jpeg(bin))
    print("Successfully loaded images: {}, labels: {}".format(
        len(images), len(labels)))

    return images, labels


def calculate_accuracy(distances, labels, threshold):
    """Calculate the true positive rate, the false positive rate and the accuracy.

    Args:
        distances: a numpy array of distances.
        labels: a numpy array of labels.
        threshold: the threshold value.

    Returns:
        the true positive rate, the false positive rate and the accuracy.
    """
    predictions = np.less(distances, threshold)

    tp = np.sum(np.logical_and(predictions, labels))
    fp = np.sum(np.logical_and(predictions, np.logical_not(labels)))

    tn = np.sum(np.logical_and(np.logical_not(predictions),
                               np.logical_not(labels)))
    fn = np.sum(np.logical_and(np.logical_not(predictions), labels))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    accuracy = float(tp + tn) / distances.size

    return tpr, fpr, accuracy


def calculate_roc(distances, labels, thresholds, num_folds=10):
    """Calculate the true positive rate, false positive rate, and accuracy.

    Args:
        distances: a numpy array of distances.
        labels: a numpy array of labels.
        threshold: a list of threshold values.

    Returns:
        the true positive rate, the false positive rate and the accuracy.
    """
    tpr_list = []
    fpr_list = []
    acc_list = []

    if num_folds == 0:
        for threshold in thresholds:
            tpr, fpr, acc = calculate_accuracy(distances, labels, threshold)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            acc_list.append(acc)
    else:
        print("Run evaluation for {} folds.".format(num_folds))

        folds = KFold(n_splits=num_folds, shuffle=False).split(
            np.arange(len(labels)))

        for train_set, test_set in folds:
            _tprs = []
            _fprs = []
            _acc = []

            for threshold in thresholds:
                _, _, acc = calculate_accuracy(
                    distances[train_set], labels[train_set], threshold)
                _acc.append(acc)

                tpr, fpr, _ = calculate_accuracy(
                    distances[test_set], labels[test_set], threshold)
                _tprs.append(tpr)
                _fprs.append(fpr)

            # Find the best threshold for the current fold.
            best_threshold = thresholds[np.argmax(_acc)]

            # Get the accuracy with the BEST threshold.
            _, _, acc = calculate_accuracy(
                distances[test_set], labels[test_set], best_threshold)

            # Summary current fold.
            tpr_list.append(_tprs)
            fpr_list.append(_tprs)
            acc_list.append(acc)

        tpr_list = np.mean(np.array(tpr_list), 0)
        fpr_list = np.mean(np.array(fpr_list), 0)

    return tpr_list, fpr_list, acc_list


def evaluate(dataset, model, batch_size, num_folds=10):
    """Evaluate model on dataset.

    Args:
        dataset: a single test dataset.
        model: the model to be evaluated.
        batch_size: inference batch size.
        num_fold: test number of folds.

    Returns:
        tpr_list: the true positive rate list.
        fpr_list: the false positive rate list.
        acc_list: the accuracy list.
        thresholds: the threshold list.
    """
    images, labels = dataset
    images = tf.data.Dataset.from_tensor_slices(images).batch(batch_size)
    embeddings = []

    print("Getting embeddings..")
    for image_batch in images:
        inputs = model._preprocess(image_batch)
        embeddings.extend(tf.unstack(model._get_embeddings(inputs)))
    print("Embeddings got.")

    embeddings_1 = embeddings[0::2]
    embeddings_2 = embeddings[1::2]

    print("Calculate distances..")
    distances = model._get_distances(embeddings_1, embeddings_2, True)
    distance_min = min(distances)
    distance_max = max(distances)
    print("Distances got. Min: {:.4f}, Max: {:.4f}".format(
        distance_min, distance_max))

    thresholds = np.arange(0, np.ceil(distance_max), 0.01)

    tpr_list, fpr_list, acc_list = calculate_roc(np.array(distances, dtype=np.float32),
                                                 np.array(labels),
                                                 thresholds,
                                                 num_folds)

    return tpr_list, fpr_list, acc_list, thresholds.tolist()


if __name__ == '__main__':
    # Loading test sets.
    test_set_name = "LFW"
    bin_lfw = "/home/robin/hdd/data/raw/face/ms1m/faces_ms1m-refine-v2_112x112/faces_emore/lfw.bin"
    image_size = [112, 112]
    if os.path.exists(bin_lfw):
        print('loading.. ', test_set_name)
        data_set = load_bin(bin_lfw, image_size)

    # Loading the inference model.
    print("Loading model..")
    ai = WanderingAI("exported/hrnetv2")

    # Run the evaluations.
    batch_size = 100
    tprs, fprs, accs, thresholds = evaluate(data_set, ai, batch_size)
    print("{} Max accuracy: {:.4f}, threshold: {}".format(
        test_set_name, max(accs), thresholds[np.argmax(accs)]))

    # Draw the ROC curve.
    plt.plot(fprs, tprs, linewidth=2.0)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.suptitle('ROC of {}'.format(test_set_name))
    plt.show()
