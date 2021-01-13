
"""Converting MXNET record into TensorFlow record.

Run this module like this:
    python3 -m utils.mx_2_tf
"""

import mxnet as mx
import tensorflow as tf
from tqdm import tqdm

from .tfrecord_operator.tfrecord_operator import RecordOperator, int64_feature, bytes_feature


class MS1M(RecordOperator):
    """Construct MS1M tfrecord files."""

    def make_example(self, image_string, label):
        """Construct an tf.Example with image data and label.
        Args:
            image_string: encoded image, NOT as numpy array.
            label: the label.
        Returns:
            a tf.Example.
        """

        image_shape = tf.image.decode_jpeg(image_string).shape

        # After getting all the features, time to generate a TensorFlow example.
        feature = {
            'image/height': int64_feature(image_shape[0]),
            'image/width': int64_feature(image_shape[1]),
            'image/depth': int64_feature(image_shape[2]),
            'image/encoded': bytes_feature(image_string),
            'label': int64_feature(label),
        }

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=feature))

        return tf_example

    def set_feature_description(self):
        self.feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }


def convert(mx_record, mx_idx, tf_record):
    # Construct a converter.
    converter = MS1M(tf_record)

    # Construct a MXNET record reader.
    print("Reading MXNET record...")
    mx_records = mx.recordio.MXIndexedRecordIO(mx_idx, mx_record, 'r')

    # Read the header to get total records count, which is embedded in the
    # first(0th) record.
    record_head = mx_records.read_idx(0)
    header, _ = mx.recordio.unpack(record_head)
    total_samples_num = int(header.label[0])
    print("Total records: {}".format(total_samples_num))

    # After getting the total count, we can loop through all of them and save
    # all examples in a TFRecord file.
    print("Converting record...")
    for i in tqdm(range(1, total_samples_num)):
        # Read a record from MXNET records with image_string and label.
        a_record = mx_records.read_idx(i)
        header, image_string = mx.recordio.unpack(a_record)
        label = int(header.label)

        # Convert the image and label to a tf.Example.
        tf_example = converter.make_example(image_string, label)

        # Write the example to file.
        converter.write_example(tf_example)

    print("All done. Record file is:\n{}".format(tf_record))


def count_samples(parsed_dataset):
    counter = 0
    for sample in parsed_dataset:
        counter += 1
    return counter


def save_one_sample_to_file(parsed_dataset, file_to_write='sample.jpg'):
    for sample in parsed_dataset:
        label = sample['label'].numpy()
        image_raw = sample['image/encoded']
        with tf.io.gfile.GFile(file_to_write, 'w') as fp:
            fp.write(image_raw.numpy())
        break
    print('One record parsed, label: {}'.format(label))
    print("An image extracted had been written to the current directory as {}".format(
        file_to_write))


if __name__ == "__main__":
    # MXNET record:
    mx_idx = '/home/robin/data/face/faces_ms1m-refine-v2_112x112/faces_emore/train.idx'
    mx_record = '/home/robin/data/face/faces_ms1m-refine-v2_112x112/faces_emore/train.rec'

    # The TFRecord file you want to generate.
    tf_record = "/home/robin/data/face/faces_ms1m-refine-v2_112x112/faces_emore/train.record"

    # Generate TFRecord file.
    convert(mx_record, mx_idx, tf_record)

    # Test the file.
    # Parse the dataset.
    ms1m = MS1M(tf_record)
    dataset = ms1m.parse_dataset()
    print("Total samples: ", count_samples(dataset))

    # Extract one sample from the record file.
    save_one_sample_to_file(dataset, 'sample.jpg')
