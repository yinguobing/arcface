"""This module provides the dataset parsing function to generate the training
and testing data."""

import tensorflow as tf


def build_dataset(tfrecord_file,
                  batch_size,
                  training=False,
                  buffer_size=65536):
    """Generate parsed TensorFlow dataset.

    Args:
        tfrecord_file: the tfrecord file path.
        batch_size: batch size.
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
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    # Now construct the dataset from tfrecord file.
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Parse the dataset to get samples.
    dataset = dataset.map(_parse_function, num_parallel_calls=autotune)

    # Shuffle the data if training.
    if training:
        dataset = dataset.shuffle(buffer_size)

    # Batch the data.
    dataset = dataset.batch(batch_size)

    # Prefetch the data to accelerate the pipeline.
    dataset = dataset.prefetch(autotune)

    return dataset
