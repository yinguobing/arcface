"""The training module for ArcFace face recognition."""

import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras

from dataset import build_dataset
from losses import ArcLoss
from network import hrnet_v2

parser = ArgumentParser()
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--initial_epoch", default=0, type=int,
                    help="From which epochs to resume training.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Training batch size.")
parser.add_argument("--export_only", default=False, type=bool,
                    help="Save the model without training.")
args = parser.parse_args()


def restore_checkpoint(checkpoint_dir, model):
    """Restore the model from checkpoint files if available.

    Args:
        checkpoint_dir: the path to the checkpoint files.
        model: the model to be restored.

    Returns:
        a boolean indicating whether the model was restored successfully.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Checkpoint directory created: {}".format(checkpoint_dir))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))
        return True
    else:
        print("Checkpoint not found. Model weights will be initialized randomly.")
        return False


if __name__ == "__main__":
    # Deep neural network training is complicated. The first thing is making
    # sure you have everything ready for training, like datasets, checkpoints,
    # logs, etc. Modify these paths to suit your needs.

    # What is the model's name?
    name = "hrnetv2"

    # Where are the training files?
    train_files = "/home/robin/data/face/faces_ms1m-refine-v2_112x112/faces_emore/train.record"

    # Where are the testing files?
    test_files = None

    # Where are the validation files? Set `None` if no files available. Then 10%
    # of the training files will be used as validation samples.
    val_files = None

    # What is the shape of the input image?
    input_shape = (112, 112, 3)

    # What is the size of the embeddings that represent the faces?
    embedding_size = 512

    # How many identities do you have in the training data?
    num_ids = 85742

    # That should be sufficient for training. However if you want more
    # customization, please keep going.

    # Checkpoint is used to resume training.
    checkpoint_dir = os.path.join("checkpoints", name)

    # Save the model for inference later.
    export_dir = os.path.join("exported", name)

    # Log directory will keep training logs like loss/accuracy curves.
    log_dir = os.path.join("logs", name)

    # All sets. Now it's time to build the model. There are two steps in ArcFace
    # training: 1, training with softmax loss; 2, training with arcloss. This
    # means not only different loss functions but also fragmented models.

    # First model is base model which output the face embeddings.
    base_model = hrnet_v2(input_shape=input_shape, output_size=embedding_size,
                          width=18, name="embedding_model")

    # Then build the second model for softmax training.
    model = keras.Sequential([keras.Input(input_shape),
                              base_model,
                              keras.layers.Dense(num_ids)],)

    # TODO: Build model for arcloss.

    # Model built. Restore the latest model if checkpoints are available.
    restored = restore_checkpoint(checkpoint_dir, model)

    # If required by user input, save the model and quit training.
    if args.export_only:
        if not restored:
            print("Warning: Model not restored from any checkpoint.")
        print("Saving model to {} ...".format(export_dir))
        model.save(export_dir)
        print("Model saved at: {}".format(export_dir))
        quit()

    # Finally, it's time to train the model.

    # Compile the model and print the model summary.
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=ArcLoss(num_ids, use_softmax=True))
    model.summary()

    # All done. The following code will setup and start the trainign.

    # Save a checkpoint. This could be used to resume training.
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, name),
        save_weights_only=True,
        verbose=1,
        save_best_only=True)

    # Visualization in TensorBoard
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                       histogram_freq=1024,
                                                       write_graph=True,
                                                       update_freq='epoch')

    # List all the callbacks.
    callbacks = [callback_checkpoint, callback_tensorboard]

    # Construct training datasets.
    dataset_train = build_dataset(train_files,
                                  batch_size=args.batch_size,
                                  training=True)

    # Construct dataset for validation. The loss value from this dataset will be
    # used to decide which checkpoint should be preserved.
    if val_files:
        dataset_val = build_dataset(val_files,
                                    batch_size=args.batch_size,
                                    training=False)
    else:
        dataset_val = dataset_train.take(int(512/args.batch_size))
        dataset_train = dataset_train.skip(int(512/args.batch_size))

    # Start training loop.
    model.fit(dataset_train,
              validation_data=dataset_val,
              epochs=args.epochs,
              callbacks=callbacks,
              initial_epoch=args.initial_epoch)
