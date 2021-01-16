"""This module provides the loss function of ArcFace"""
import tensorflow as tf
from math import pi


class ArcLoss(tf.keras.losses.Loss):
    """Additive angular margin loss.

    Original implementation: https://github.com/deepinsight/insightface
    """

    def __init__(self, margins=(1.0, 0.5, 0.0), scale=64, name="arcloss"):
        """Build an additive angular margin loss object for Keras model."""
        super().__init__(name=name)
        self.m1 = margins[0]
        self.m2 = margins[1]
        self.m3 = margins[2]
        self.scale = scale

    def call(self, y_true, y_pred):
        mapping_label_onehot = y_true
        fc7 = y_pred

        if self.m1 == 1.0 and self.m2 == 0.0:
            _one_hot = mapping_label_onehot * self.m3
            fc7 = fc7 - _one_hot
        else:
            fc7_onehot = fc7 * mapping_label_onehot
            t = tf.math.acos(fc7_onehot)
            fc7_margin = tf.math.cos(t * self.m1 + self.m2) - self.m3
            fc7_margin_onehot = fc7_margin * mapping_label_onehot
            diff = fc7_margin_onehot - fc7_onehot
            fc7 = fc7 + diff

        fc7 = fc7 * self.scale

        loss = tf.nn.softmax_cross_entropy_with_logits(y_true, fc7)

        return loss
