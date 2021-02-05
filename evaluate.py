"""Evaluation on the Labeled Faces in the Wild dataset 

This file is modified from the ArcFace official implementation in order to run
with TensorFlow. You can find the original file here:
https://github.com/deepinsight/insightface/blob/master/recognition/ArcFace/verification.py
"""

import os
import pickle

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


def calculate_folded_accuracy(distances, labels, threshold, num_folds=10):
    """Calculate the folded true positive rate, false positive rate, and accuracy.

    Args:
        distances: a numpy array of distances.
        labels: a numpy array of labels.
        threshold: the threshold value.

    Returns:
        the true positive rate, the false positive rate and the accuracy.
    """

    labels = np.asarray(labels)
    num_pairs = len(labels)
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    tprs = np.zeros((num_folds, num_thresholds))
    fprs = np.zeros((num_folds, num_thresholds))

    accuracy = np.zeros((num_folds))
    indices = np.arange(num_pairs)

    print("Run evaluation for {} folds.".format(num_folds))
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print("Fold {}".format(fold_idx))
        # Find the best threshold for the current fold.
        acc_train = np.zeros((num_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        print('threshold', thresholds[best_threshold_index])

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, distances[test_set],
                labels[test_set])

        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], distances[test_set],
            labels[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def evaluate(dataset, model, batch_size, num_folds=10):
    """Evaluate model on dataset.

    Args:
        dataset: a single test dataset.
        model: the model to be evaluated.
        batch_size: inference batch size.
        num_fold: test number of folds.

    Returns:
        the accuracy and the std.
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
    tpr_list = []
    fpr_list = []
    acc_list = []
    for threshold in thresholds:
        tpr, fpr, acc = calculate_accuracy(np.array(distances, dtype=np.float32),
                                           np.array(labels), threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        acc_list.append(acc)

    return tpr_list, fpr_list, acc_list


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
    tprs, fprs, accs = evaluate(data_set, ai, batch_size)
    print("{} Max accuracy: {:.4f}".format(test_set_name, max(accs)))
