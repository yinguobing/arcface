"""The training module for ArcFace face recognition."""

import os
from tensorflow.keras import metrics
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
parser.add_argument("--restore_weights_only", default=False, type=bool,
                    help="Only restore the model weights from checkpoint.")
args = parser.parse_args()


def restore_checkpoint(checkpoint, manager, model, weights_only=False):
    """Restore the model from checkpoint files if available.

    Args:
        checkpoint: the checkpoint defining the objects saved.
        manager: the checkpoint manager.
        model: the model to be restored.
        weights_only: only restore the model weights if set to True.
    """
    latest_checkpoint = manager.latest_checkpoint
    if latest_checkpoint:
        print("Checkpoint found: {}".format(latest_checkpoint))
    else:
        print("WARNING: Checkpoint not found. Model will be initialized from scratch.")

    if weights_only:
        checkpoint = tf.train.Checkpoint(model)
        print("Only the model weights will be restored.")

    print("Restoring..")
    checkpoint.restore(manager.latest_checkpoint)
    print("Checkpoint restored: {}".format(latest_checkpoint))


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

    # Update the metrics.
    metric_train_acc.update_state(y_batch, logits)
    metric_train_loss.update_state(loss_value)

    return loss_value


def reset_metrics():
    """Reset training metrics."""
    metric_train_acc.reset_states()
    metric_train_loss.reset_states()


def _log_n_checkpoint():
    """Log and checkpoint the model.

    Since checkpint and logging occupy lines of code and run frequently,
     define a function to make the code concise.
     """
    current_step = int(checkpoint.step)

    # Is current model the best one we had ever seen?
    best_model_found = True if (
        metric_train_loss.result() < checkpoint.last_monitor_value) else False

    # Update the checkpoint before saving.
    checkpoint.last_monitor_value.assign(
        metric_train_loss.result())

    # Log the training progress to TensorBoard..
    with summary_writer_train.as_default():
        tf.summary.scalar("loss", metric_train_loss.result(),
                          step=current_step)
        tf.summary.scalar("accuracy", metric_train_acc.result(),
                          step=current_step)
        tf.summary.scalar("learning rate",
                          optimizer._decayed_lr('float32'),
                          step=current_step)

    # ..and STDOUT.
    print("Training accuracy: {:.4f}, mean loss: {:.2f}".format(
        float(metric_train_acc.result()),
        float(metric_train_loss.result())))

    # If the best model found, save it.
    if best_model_found:
        model_scout.save()
        print("Best model found and saved.")

    # Save a regular checkpoint.
    reset_metrics()
    ckpt_manager.save()
    print("Checkpoint saved for step {}".format(current_step))


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
    frequency = 1000

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

    # Construct an optimizer. This optimizer is different from the official
    # implementation which use SGD with momentum.
    optimizer = keras.optimizers.Adam()

    # Construct the metrics for the model.
    metric_train_acc = keras.metrics.CategoricalAccuracy(name="train_accuracy")
    metric_train_loss = keras.metrics.Mean(name="train_loss_mean",
                                           dtype=tf.float32)

    # User summary writer to log the training process to TensorBoard.
    summary_writer_train = tf.summary.create_file_writer(
        os.path.join(log_dir, "train"))

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

    # The best model may not always manifest at the last training step. It is
    # better if we can track one besides regular checkpoints,
    model_scout = tf.train.CheckpointManager(
        checkpoint, os.path.join(checkpoint_dir, "model_scout"), 1)

    # Restore the latest model if checkpoints are available.
    restore_checkpoint(checkpoint, ckpt_manager, model,
                       args.restore_weights_only)

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

    # Safety check. Make sure the epochs is larger than that of the checkpoint.
    assert initial_epoch <= args.epochs, "Total epoch number {} should be \
        larger than {} of the checkpoint.".format(args.epochs, initial_epoch)

    # Start training loop.
    for epoch in range(initial_epoch, args.epochs + 1):
        # Make the epoch number human friendly.
        print("\nEpoch {}/{}".format(epoch, args.epochs))

        # Visualize the training progress.
        progress_bar = tqdm(total=steps_per_epoch, initial=initial_step,
                            ascii="->", colour='#1cd41c')

        # Iterate over the batches of the dataset
        for x_batch, y_batch in checkpoint.dataset:

            # Train for one step.
            loss = train_step(x_batch, y_batch)

            # Update the checkpoint.
            checkpoint.step.assign_add(1)

            # Update the progress bar.
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": loss.numpy(),
                "accuracy": metric_train_acc.result().numpy()})

            # Log and checkpoint the model.
            if int(checkpoint.step) % frequency == 0:
                _log_n_checkpoint()

        # Update the checkpoint epoch counter.
        checkpoint.last_epoch.assign_add(1)

        # Reset the training dataset.
        checkpoint.dataset = iter(dataset_train)

        # Save the last checkpoint.
        _log_n_checkpoint()

        # Clean up the progress bar.
        progress_bar.close()

    print("Training accomplished at epoch {}".format(args.epochs))
