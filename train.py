"""The training module for ArcFace face recognition."""

import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras

from dataset import build_dataset
from losses import ArcLoss
from network import hrnet_v2, ArcLayer

parser = ArgumentParser()
parser.add_argument("--softmax", default=False, type=bool,
                    help="Training with softmax loss.")
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--initial_epoch", default=0, type=int,
                    help="From which epochs to resume training.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Training batch size.")
parser.add_argument("--steps_per_epoch", default=512, type=int,
                    help="The number of steps for each epoch.")
parser.add_argument("--skip_data_steps", default=0, type=int,
                    help="The number of steps to skip for dataset.")
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
        tf.train.Checkpoint(model).restore(os.path.join(checkpoint_dir, name))
        print("Checkpoint restored: {}".format(latest_checkpoint))
        return True
    else:
        print("WARNING: Checkpoint not found. Model weights will be initialized randomly.")
        return False


def export(model, export_dir):
    """Export the model in saved_model format.

    Args:
        model: the keras model to be saved.
        export_dir: the direcotry where the model will be saved.
    """
    print("Saving model to {} ...".format(export_dir))
    model.save(export_dir)
    print("Model saved at: {}".format(export_dir))


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

    # Where are the validation files? Set `None` if no files available.
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

    # How many steps are there in one epoch?
    steps_per_epoch = args.steps_per_epoch

    # All sets. Now it's time to build the model. There are two steps in ArcFace
    # training: 1, training with softmax loss; 2, training with arcloss. This
    # means not only different loss functions but also fragmented models.

    # First model is base model which outputs the face embeddings.
    base_model = hrnet_v2(input_shape=input_shape, output_size=embedding_size,
                          width=18, name="embedding_model")

    # Then build the second model for training.
    if args.softmax:
        print("Building training model with softmax loss...")
        model = keras.Sequential([keras.Input(input_shape),
                                  base_model,
                                  keras.layers.Dense(num_ids),
                                  keras.layers.Softmax()],
                                 name="training_model")
        loss_fun = keras.losses.CategoricalCrossentropy()
    else:
        print("Building training model with Arc loss...")
        model = keras.Sequential([keras.Input(input_shape),
                                  base_model,
                                  ArcLayer(num_ids)],
                                 name="training_model")
        loss_fun = ArcLoss()

    # Model built. Restore the latest model if checkpoints are available.
    restore_checkpoint(checkpoint_dir, model)

    # If required by user input, save the model and quit training.
    if args.export_only:
        export(base_model, export_dir)
        quit()

    # Finally, it's time to train the model.

    # Compile the model and print the model summary.
    model.compile(optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.CategoricalAccuracy()],
                  loss=loss_fun)
    model.summary()

    # All done. The following code will setup and start the training.

    # Save a checkpoint. This could be used to resume training.
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, name),
        monitor='categorical_accuracy',
        save_weights_only=True,
        verbose=1,
        save_best_only=True)

    # Visualization in TensorBoard
    callback_tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=True)

    # List all the callbacks.
    callbacks = [callback_checkpoint, callback_tensorboard]

    # Construct training datasets.
    dataset_train = build_dataset(train_files,
                                  batch_size=args.batch_size,
                                  one_hot_depth=num_ids,
                                  training=True,
                                  buffer_size=4096)

    # Construct dataset for validation. The loss value from this dataset will be
    # used to decide which checkpoint should be preserved.
    if val_files:
        dataset_val = build_dataset(val_files,
                                    batch_size=args.batch_size,
                                    one_hot_depth=num_ids,
                                    training=False,
                                    buffer_size=4096)
    else:
        dataset_val = None

    # The MS1M dataset contains millions of image samples. If training was
    # frequently interupted, the next training loop will always restart with
    # same training date from the dataset begining. To avoid this, skip adequate
    # training samples when resume training.
    if args.skip_data_steps != 0:
        dataset_train = dataset_train.skip(args.skip_data_steps)
        print("Skipping data steps previously encountered: {}".format(
            args.skip_data_steps))

    # Start training loop.
    model.fit(dataset_train,
              validation_data=dataset_val,
              steps_per_epoch=steps_per_epoch,
              epochs=args.epochs,
              callbacks=callbacks,
              initial_epoch=args.initial_epoch)
