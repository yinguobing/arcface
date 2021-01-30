"""This module provides the implementation of training supervisor."""

import os
from numpy.lib.npyio import save

import tensorflow as tf
from tqdm import tqdm


class TrainingSupervisor(object):
    """A training supervisor will organize and monitor the training process."""

    def __init__(self, model, optimizer, loss, dataset, training_dir, save_freq, monitor) -> None:
        """Training supervisor organizes and monitors the training process.

        Args:
            model: the Keras model to be trained.
            optimizer: a Keras optimizer used for training.
            loss: a Keras loss function.
            dataset: the training dataset.
            training_dir: the directory to save the training files.
            save_freq: integer, the supervisor saves the model at end of this many batches.
            monitor: the metric name to monitor.
            mode: one of {'min', 'max'}
        """
        super().__init__()

        # Track the objects used for training.
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss
        self.dataset = dataset
        self.save_freq = save_freq
        self.metrics = {
            'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(
                name='train_accuracy', dtype=tf.float32),
            'loss': tf.keras.metrics.Mean(name="train_loss_mean",
                                          dtype=tf.float32)}
        self.monitor = self.metrics[monitor]

        # Training schedule tracks the training progress. The training
        # supervisor uses this object to make training arrangement. The schedule
        # is saved in the checkpoint and maintained by the manager.
        self.schedule = {
            'step': tf.Variable(0, trainable=False, dtype=tf.int64),
            'epoch': tf.Variable(1, trainable=False, dtype=tf.int64),
            'monitor_value': tf.Variable(0, trainable=False, dtype=tf.float32)}

        # Both the model and the training status shall be tracked. A TensorFlow
        # checkpoint is the best option to fullfill this job.
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            loss_fun=self.loss_fun,
            metrics=self.metrics,
            schedule=self.schedule,
            monitor=self.monitor)

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

    @tf.function
    def _train_step(self, x_batch, y_batch):
        """Define the training step function.

        Args:
            x_batch: the inputs of the network.
            y_batch: the labels of the batched inputs.

        Returns:
            logtis and loss.
        """

        with tf.GradientTape() as tape:
            # Run the forward propagation.
            logits = self.model(x_batch, training=True)

            # Calculate the loss value from targets and regularization.
            loss = self.loss_fun(y_batch, logits) + sum(self.model.losses)

        # Calculate the gradients.
        grads = tape.gradient(loss, self.model.trainable_weights)

        # Back propagation.
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        return logits, loss

    @tf.function
    def _update_metrics(self, labels, logits, loss):
        """Update the metrics.

        Args:
            labels: the labels of the batched inputs.
            logits: the outputs of the model.
            loss: the loss value of current training step.
        """
        self.metrics['categorical_accuracy'].update_state(labels, logits)
        self.metrics['loss'].update_state(loss)

    def _reset_metrics(self):
        """Reset all the metrics."""
        for _, metric in self.metrics.items():
            metric.reset()

    def _log_to_tensorboard(self):
        """Log the training process to TensorBoard."""
        # Get the parameters to log.
        current_step = int(self.schedule['step'])
        train_loss = self.metrics['loss'].result()
        train_acc = self.metrics['categorical_accuracy'].result()
        lr = self.optimizer._decayed_lr('float32')

        with self.clerk.as_default():
            tf.summary.scalar("loss", train_loss,   step=current_step)
            tf.summary.scalar("accuracy", train_acc, step=current_step)
            tf.summary.scalar("learning rate", lr, step=current_step)

        # Log to STDOUT.
        print("Training accuracy: {:.4f}, mean loss: {:.2f}".format(
            float(train_acc), float(train_loss)))
