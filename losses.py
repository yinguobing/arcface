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
            cos_t = fc7_onehot
            t = tf.math.arccos(cos_t)

            if self.m1 != 1.0:
                t = t * self.m1

            if self.m2 != 0.0:
                t = t + self.m2

            margin_cos = tf.math.cos(t)
            if self.m3 != 0.0:
                margin_cos = margin_cos - self.m3

            margin_fc7 = margin_cos
            margin_fc7_onehot = margin_fc7 * mapping_label_onehot
            diff = margin_fc7_onehot - fc7_onehot

            fc7 = fc7 + diff

        fc7 = fc7 * self.scale

        return fc7
