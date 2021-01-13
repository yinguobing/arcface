"""This module provides the loss function of ArcFace"""
import tensorflow as tf
from tensorflow.python.ops import losses


class ArcLoss(tf.keras.losses.Loss):
    """Additive angular margin loss."""

    def __init__(self, num_ids, use_softmax=False, name="arcloss"):
        """Build an additive angular margin loss object for Keras model.

        Args:
            num_ids: number of identities.
            use_softmax: output softmax values for loss.
        """
        super().__init__(name=name)
        self.use_softmax = use_softmax
        self.num_ids = num_ids

    def call(self, y_true, y_pred):
        if self.use_softmax:
            y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32),
                                depth=self.num_ids,
                                dtype=tf.float32)
            
            losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        else:
            # TODO: implement arcloss.
            pass

        return losses
