"""The training module for ArcFace face recognition."""

import os
from tqdm import tqdm
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras

from dataset import build_dataset
from losses import ArcLoss
from network import L2Normalization, hrnet_v2, ArcLayer

parser = ArgumentParser()
parser.add_argument("--softmax", default=False, type=bool,
                    help="Training with softmax loss.")
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Training batch size.")
parser.add_argument("--skip_data_steps", default=0, type=int,
                    help="The number of steps to skip for dataset.")
parser.add_argument("--export_only", default=False, type=bool,
                    help="Save the model without training.")
args = parser.parse_args()


def restore_checkpoint(checkpoint, manager):
    """Restore the model from checkpoint files if available.

    Args:
        checkpoint: the checkpoint.
        manager: the checkpoint manager.

    Returns:
        a boolean indicating whether the model was restored successfully.
    """
    latest_checkpoint = manager.latest_checkpoint
    if latest_checkpoint:
        print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
        checkpoint.restore(manager.latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))
    else:
        print("WARNING: Checkpoint not found. Model will be initialized from scratch.")


def export(model, export_dir):
    """Export the model in saved_model format.

    Args:
        model: the keras model to be saved.
        export_dir: the direcotry where the model will be saved.
    """
    print("Saving model to {} ...".format(export_dir))
    model.save(export_dir)
    print("Model saved at: {}".format(export_dir))


@tf.function
def train_step(x_batch, y_batch):
    """Define the training step function."""

    with tf.GradientTape() as tape:
        # Run the forward propagation.
        logits = model(x_batch, training=True)

        # Calculate the loss value from targets and regularization.
        loss_value = loss_fun(y_batch, logits) + sum(model.losses)

    # Calculate the gradients.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Back propagation.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Calculate the accuracies.
    metric_acc_train.update_state(y_batch, logits)

    return loss_value


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

    # How many identities do you have in the training dataset?
    num_ids = 85742

    # How many examples do you have in the training dataset?
    num_examples = 5822653

    # That should be sufficient for training. However if you want more
    # customization, please keep going.

    # Checkpoint is used to resume training.
    checkpoint_dir = os.path.join("checkpoints", name)

    # Save the model for inference later.
    export_dir = os.path.join("exported", name)

    # Log directory will keep training logs like loss/accuracy curves.
    log_dir = os.path.join("logs", name)

    # Any weight regularization?
    regularizer = keras.regularizers.L2(5e-4)

    # How often do you want to log and save the model, in steps?
    frequency = 100

    # All sets. Now it's time to build the model. There are two steps in ArcFace
    # training: 1, training with softmax loss; 2, training with arcloss. This
    # means not only different loss functions but also fragmented models.

    # First model is base model which outputs the face embeddings.
    base_model = hrnet_v2(input_shape=input_shape, output_size=embedding_size,
                          width=18, trainable=True,
                          kernel_regularizer=regularizer,
                          name="embedding_model")

    # Then build the second model for training.
    if args.softmax:
        print("Building training model with softmax loss...")
        model = keras.Sequential([keras.Input(input_shape),
                                  base_model,
                                  keras.layers.Dense(num_ids,
                                                     kernel_regularizer=regularizer),
                                  keras.layers.Softmax()],
                                 name="training_model")
        loss_fun = keras.losses.CategoricalCrossentropy()
    else:
        print("Building training model with ARC loss...")
        model = keras.Sequential([keras.Input(input_shape),
                                  base_model,
                                  L2Normalization(),
                                  ArcLayer(num_ids, regularizer)],
                                 name="training_model")
        loss_fun = ArcLoss()

    # Construct an optimizer with learning rate schedule. We will follow the
    # official instructions.
    schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[2e5, 3.2e5, 3.6e5],
        values=[0.1, 0.01, 0.001, 0.0001])
    optimizer = keras.optimizers.SGD(schedule, 0.9)

    # Construct the metrics for the model.
    metric_acc_train = keras.metrics.CategoricalAccuracy()

    # Construct training datasets.
    dataset_train = build_dataset(train_files,
                                  batch_size=args.batch_size,
                                  one_hot_depth=num_ids,
                                  training=True,
                                  buffer_size=4096)

    # Construct dataset for validation. The loss value from this dataset can be
    # used to decide which checkpoint should be preserved.
    if val_files:
        dataset_val = build_dataset(val_files,
                                    batch_size=args.batch_size,
                                    one_hot_depth=num_ids,
                                    training=False,
                                    buffer_size=4096)
    else:
        dataset_val = None

    # Save a checkpoint. This could be used to resume training.
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     last_epoch=tf.Variable(1),
                                     last_monitor_value=tf.Variable(0.0),
                                     optimizer=optimizer,
                                     model=model,
                                     dataset=iter(dataset_train))
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, 2)

    # Restore the latest model if checkpoints are available.
    restore_checkpoint(checkpoint, ckpt_manager)

    # If training accomplished, save the base model for inference.
    if args.export_only:
        export(base_model, export_dir)
        quit()

    # If training shall be resumed, where are we now?
    global_step = checkpoint.step.numpy()
    steps_per_epoch = num_examples // args.batch_size
    initial_step = global_step % steps_per_epoch
    initial_epoch = checkpoint.last_epoch.numpy()
    print("Resume training from global step: {}, epoch: {}".format(
        global_step, initial_epoch))

    # Start training loop.
    epochs = args.epochs - initial_epoch

    for epoch in range(epochs):
        # Make the epoch number human friendly.
        epoch += initial_epoch
        print("\nEpoch {}/{}".format(epoch, args.epochs))

        # Visualize the training progress.
        progress_bar = tqdm(total=steps_per_epoch, initial=initial_step,
                            ascii="->", colour='#1cd41c')

        # Iterate over the batches of the dataset
        for x_batch, y_batch in checkpoint.dataset:

            # Train for one step.
            loss = train_step(x_batch, y_batch)

            # Update the monitor value.
            checkpoint.last_monitor_value.assign(loss)

            # Update the checkpoint step counter.
            checkpoint.step.assign_add(1)

            # Update the progress bar.
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.numpy()})

            # Log and checkpoint the model.
            if int(checkpoint.step) % frequency == 0:
                # Log the training progress.
                train_acc = metric_acc_train.result()
                print("CTraining accuracy: {:.4f}".format(float(train_acc)))

                # Save the checkpoint.
                ckpt_manager.save()
                print("Checkpoint saved for step {}".format(int(checkpoint.step)))

        # Update the checkpoint epoch counter.
        checkpoint.last_epoch.assign_add(1)

        # Reset training metrics at the end of each epoch
        metric_acc_train.reset_states()

        # Clean up the progress bar.
        progress_bar.close()
