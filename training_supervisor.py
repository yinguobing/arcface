"""This module provides the implementation of training supervisor."""

import os

import tensorflow as tf
from tqdm import tqdm


class TrainingSupervisor(object):
    """A training supervisor will organize and monitor the training process."""

    def __init__(self, model, optimizer, loss, metrics, dataset, training_dir) -> None:
        """Training supervisor organizes and monitors the training process.

        Args:
            model: the Keras model to be trained.
            optimizer: a Keras optimizer used for training.
            loss: a Keras loss function.
            metrics: List of metrics to be evaluated during training.
            dataset: the training dataset.
            training_dir: the directory to save the training files.
        """
        super().__init__()
        # Track the objects used for training.
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss
        self.dataset = dataset
        self.metrics = {
            'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(
                name='train_accuracy', dtype=tf.float32),
            'loss': tf.keras.metrics.Mean(name="train_loss_mean",
                                          dtype=tf.float32)}

        # Training schedule tracks the training progress. The training
        # supervisor uses this object to make training arrangement. The schedule
        # is saved in the checkpoint and maintained by the manager.
        self.schedule = {
            'step': tf.Variable(0, trainable=False, dtype=tf.int64),
            'epoch': tf.Variable(1, trainable=False, dtype=tf.int64)}

        # Both the model and the training status shall be tracked. A TensorFlow
        # checkpoint is the best option to fullfill this job.
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            loss_fun=self.loss_fun,
            metrics=self.metrics,
            schedule=self.schedule)

        # A model manager is responsible for saving the current training
        # schedule and the model weights.
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            os.path.join(training_dir, 'checkpoints'),
            max_to_keep=2)

        # A model scout watches and saves the best model according to the
        # monitor value.
        self.scout = tf.train.CheckpointManager(
            self.checkpoint,
            os.path.join(training_dir, 'model_scout'),
            max_to_keep=1)

        # A clerk writes the training logs to the TensorBoard.
        self.clerk = tf.summary.create_file_writer(
            os.path.join(training_dir, 'logs'))

    def restore(self, weights_only=False):
        """Restore training process from previous training checkpoint.

        Args:
            weights_only: only restore the model weights. Default is False.
        """
        # Are there any checkpoint files?
        latest_checkpoint = self.manager.latest_checkpoint

        if latest_checkpoint:
            print("Checkpoint found: {}".format(latest_checkpoint))
        else:
            print("WARNING: Checkpoint not found. Model will be initialized \
                from scratch.")

        print("Restoring..")

        if weights_only:
            print("Only the model weights will be restored.")
            checkpoint = tf.train.Checkpoint(self.model)
            checkpoint.restore(latest_checkpoint)
        else:
            self.checkpoint.restore(latest_checkpoint)

        print("Checkpoint restored: {}".format(latest_checkpoint))
