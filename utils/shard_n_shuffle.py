"""Fully shuffle the samples in the TFRecord file.

The shuffling process required adequate free spaces that should be more than 
double of the TFRecord file size. Keep this in mind or you may get a incomplete
dataset compared with the original one.
"""
import numpy as np
import tensorflow as tf
import time
import os


def shuffle_once(input_file, output_file):
    """Make a new record file with shuffled samples from the input record file.

    Args:
        record_file: the target file to be shuffled.
        rounds: number of rounds the file will be shuffled.

    Returns:
        the shuffled record file path.
    """
    # Construct TFRecord file writer.
    writer = tf.io.TFRecordWriter(output_file)

    # Read in the dataset.
    dataset = tf.data.TFRecordDataset(input_file)

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
    starting = time.time()

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

    print("Total samples: {}".format(counter))
    print("Elapsed time: {}".format(
        time.strftime("%H:%M:%S", time.gmtime(time.time()-starting))))

    writer.close()


def shuffle(record_file, rounds=1):
    """Shuffle the record file N times.

    Args:
        record_file: the target file to be shuffled.
        rounds: number of rounds the file will be shuffled.
    """
    assert(rounds >= 1, "Rounds should be larger than or equal to 1.")

    output_file = record_file.rpartition('.')[0]+'_shuffled.record'
    temp_pair = [record_file.rpartition('.')[0]+'_0.tmp',
                 record_file.rpartition('.')[0]+'_1.tmp']

    for round in range(rounds):
        print("Round {} of total {}".format(round, rounds))

        if round == 1:
            shuffle_once(record_file, temp_pair[1])
        else:
            temp_pair.reverse()
            shuffle_once(temp_pair[0], temp_pair[1])

    os.rename(temp_pair[1], output_file)
    if rounds > 1:
        os.remove(temp_pair[0])


if __name__ == "__main__":
    # Where is the TFRecord file to be shuffled?
    record_file = "/home/robin/data/face/faces_ms1m-refine-v2_112x112/faces_emore/train_0.record"

    # How many times the file should be shuffled?
    rounds = 3

    # Shuffling..
    shuffle(record_file, rounds)

    print("All done.")
