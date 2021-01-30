"""This module provides the implementation of training supervisor."""

import os

import tensorflow as tf
from tqdm import tqdm


class TrainingSupervisor(object):
    """A training supervisor will organize and monitor the training process."""

    def __init__(self, model, optimizer, loss, dataset, training_dir, save_freq, monitor, mode, name) -> None:
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
            name: current model or project name.
        """
        super().__init__()

        # Track the objects used for training.
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss
        self.dataset = dataset
        self.data_generator = iter(self.dataset)
        self.save_freq = save_freq
        self.metrics = {
            'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(
                name='train_accuracy', dtype=tf.float32),
            'loss': tf.keras.metrics.Mean(name="train_loss_mean",
                                          dtype=tf.float32)}
        self.monitor = self.metrics[monitor]
        self.mode = mode

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
            metrics=self.metrics,
            schedule=self.schedule,
            monitor=self.monitor,
            dataset=self.data_generator)

        # A model manager is responsible for saving the current training
        # schedule and the model weights.
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            os.path.join(training_dir, 'checkpoints', name),
            max_to_keep=2)

        # A model scout watches and saves the best model according to the
        # monitor value.
        self.scout = tf.train.CheckpointManager(
            self.checkpoint,
            os.path.join(training_dir, 'model_scout', name),
            max_to_keep=1)

        # A clerk writes the training logs to the TensorBoard.
        self.clerk = tf.summary.create_file_writer(
            os.path.join(training_dir, 'logs', name))

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
            metric.reset_states

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

    def _checkpoint(self):
        """Checkpoint the current training process.

        Args:
        monitor: the metric value to monitor.
        mode: one of {'min', 'max'}
        """
        # A helper function to check values by mode.
        def _check_value(v1, v2, mode):
            if (v1 < v2) & (mode == 'min'):
                return True
            elif (v1 > v2) & (mode == 'max'):
                return True
            else:
                return False

        # Get previous and current monitor values.
        previous = self.schedule['monitor_value'].numpy()
        current = self.monitor.result()

        # For the first checkpoint, initialize the monitor value to make
        # subsequent comparisons valid.
        if previous == 0.0:
            self.schedule['monitor_value'].assign(current)

        # Is current model the best one we had ever seen?
        if _check_value(current, previous, self.mode):
            print("Monitor value improved from {:.4f} to {:.4f}."
                  .format(previous, current))

            # Update the schedule.
            self.schedule['monitor_value'].assign(current)

            # And save the model.
            best_model_path = self.scout.save()
            print("Best model found and saved: {}".format(best_model_path))
        else:
            print("Monitor value not improved: {:.4f}, latest: {:.4f}."
                  .format(previous, current))

        # Save a regular checkpoint.
        self._reset_metrics()
        ckpt_path = self.manager.save()
        print("Checkpoint saved at global step: {}, to file: {}".format(
            int(self.schedule['step']), ckpt_path))

    def train(self, epochs, steps_per_epoch):
        """Train the model for epochs.

        Args:
            epochs: an integer number of epochs to train the model.
            steps_per_epoch: an integer numbers of steps for one epoch.
        """
        # In case the training is resumed, where are now?
        initial_epoch = self.schedule['epoch'].numpy()
        global_step = self.schedule['step'].numpy()
        initial_step = global_step % steps_per_epoch

        print("Resume training from global step: {}, epoch: {}".format(
            global_step, initial_epoch))
        print("Current step is: {}".format(initial_step))

        # Start training loop.
        for epoch in range(initial_epoch, epochs + 1):
            # Log current epoch.
            print("\nEpoch {}/{}".format(epoch, epochs))

            # Visualize the training progress.
            progress_bar = tqdm(total=steps_per_epoch, initial=initial_step,
                                ascii="->", colour='#1cd41c')

            # Iterate over the batches of the dataset
            for x_batch, y_batch in self.data_generator:

                # Train for one step.
                logits, loss = self._train_step(x_batch, y_batch)

                # Update the metrics.
                self._update_metrics(y_batch, logits, loss)

                # Update the training schedule.
                self.schedule['step'].assign_add(1)

                # Update the progress bar.
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": "{:.2f}".format(loss.numpy()),
                    "accuracy": "{:.3f}".format(
                        self.metrics['categorical_accuracy'].result().numpy())})

                # Log and checkpoint the model.
                if int(self.schedule['step']) % self.save_freq == 0:
                    self._log_to_tensorboard()
                    self._checkpoint()

            # Update the checkpoint epoch counter.
            self.schedule['epoch'].assign_add(1)

            # Reset the training dataset.
            self.data_generator = iter(self.dataset)

            # Save the last checkpoint.
            self._log_to_tensorboard()
            self._checkpoint()

            # Clean up the progress bar.
            progress_bar.close()

        print("Training accomplished at epoch {}".format(epochs))

    def export(self, model, export_dir):
        """Export the model in saved_model format.

        Args:
            export_dir: the direcotry where the model will be saved.
        """
        print("Saving model to {} ...".format(export_dir))
        model.save(export_dir)
        print("Model saved at: {}".format(export_dir))

    def override(self, step=None, epoch=None, monitor_value=None):
        """Override the current training schedule with a new one.

        The parameter won't be overridden if new value is None.

        Args:
            step: new training step to start from.
            epoch: new epoch to start from.
            monitor_value: new monitor value to start with.
        """
        if step:
            self.schedule['step'].assign(step)

        if epoch:
            self.schedule['epoch'].assign(epoch)

        if monitor_value:
            self.schedule['monitor_value'].assign(monitor_value)
