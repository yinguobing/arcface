"""Fully shuffle the sample in the TFRecord file."""
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    # Setup file paths.
    record_file = "/home/robin/data/face/faces_ms1m-refine-v2_112x112/faces_emore/train.record"
    shuffled_file = record_file.rpartition('.')[0]+'_shuffled.record'

    # Construct TFRecord file writer.
    writer = tf.io.TFRecordWriter(shuffled_file)

    # Read in the dataset.
    dataset = tf.data.TFRecordDataset(record_file)

    # Evenly split the dataset shards.
    num_shards = np.random.randint(100, 300)
    buffer_size = np.random.randint(128, 256)
    shards = [iter(dataset.shard(num_shards, n).shuffle(buffer_size).prefetch(tf.data.experimental.AUTOTUNE))
              for n in range(num_shards)]

    # Loop through every shard and collect samples shuffled.
    print("Number of shards: {}".format(num_shards))
    print("Shuffle buffer size: {}".format(buffer_size))

    mini_batch_indices = np.arange(num_shards)
    counter = 0

    while True:
        if mini_batch_indices.size == 0:
            break

        np.random.shuffle(mini_batch_indices)

        for index in mini_batch_indices:
            try:
                example = shards[index].get_next()
            except Exception:
                mini_batch_indices = np.setdiff1d(
                    mini_batch_indices, np.array(index))
                print("Shard [{}] exhausted, left {} shards.".format(
                    index, mini_batch_indices.size))
                break
            else:
                writer.write(example.numpy())
                counter += 1
                print("Sample processed: {}".format(counter), "\033[1A")

    print("Total samples; {}".format(counter))

    writer.close()
