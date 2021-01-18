"""This module provides the dataset parsing function to generate the training
and testing data."""

import tensorflow as tf
from preprocessing import normalize


def build_dataset(tfrecord_file,
                  batch_size,
                  one_hot_depth,
                  training=False,
                  buffer_size=4096):
    """Generate parsed TensorFlow dataset.

    Args:
        tfrecord_file: the tfrecord file path.
        batch_size: batch size.
        one_hot_depth: the depth for one hot encoding, usually the number of 
            classes.
        training: a boolean indicating whether the dataset will be used for
            training.
        buffer_size: hwo large the buffer is for shuffling the samples.

    Returns:
        a parsed dataset.
    """
    # Let TensorFlow tune the input pipeline automatically.
    autotune = tf.data.experimental.AUTOTUNE

    # Describe how the dataset was constructed. The author who created the file
    # is responsible for this information.
    feature_description = {'image/height': tf.io.FixedLenFeature([], tf.int64),
                           'image/width': tf.io.FixedLenFeature([], tf.int64),
                           'image/depth': tf.io.FixedLenFeature([], tf.int64),
                           'image/encoded': tf.io.FixedLenFeature([], tf.string),
                           'label': tf.io.FixedLenFeature([], tf.int64)}

    # Define a helper function to decode the tf-example. This function will be
    # called by map() later.
    def _parse_function(example):
        features = tf.io.parse_single_example(example, feature_description)
        image = tf.image.decode_jpeg(features['image/encoded'])
        image = normalize(image)
        label = tf.one_hot(features['label'], depth=one_hot_depth,
                           dtype=tf.float32)

        return image, label

    # Now construct the dataset from tfrecord file and make it indefinite.
    dataset = tf.data.TFRecordDataset(tfrecord_file).repeat()

    # Shuffle the data if training.
    if training:
        dataset = dataset.shuffle(buffer_size)

    # Parse the dataset to get samples.
    dataset = dataset.map(_parse_function, num_parallel_calls=autotune)

    # Batch the data.
    dataset = dataset.batch(batch_size)

    # Prefetch the data to accelerate the pipeline.
    dataset = dataset.prefetch(autotune)

    return dataset
