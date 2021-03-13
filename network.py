"""This module provides the network backbone implementation."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class L2Normalization(keras.layers.Layer):
    """This layer normalizes the inputs with l2 normalization."""

    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        inputs = tf.nn.l2_normalize(inputs, axis=1)

        return inputs

    def get_config(self):
        config = super().get_config()
        return config


class ArcLayer(keras.layers.Layer):
    """Custom layer for ArcFace.

    This layer is equivalent a dense layer except the weights are normalized.
    """

    def __init__(self, units, kernel_regularizer=None, **kwargs):
        super(ArcLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-1], self.units],
                                      dtype=tf.float32,
                                      initializer=keras.initializers.HeNormal(),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name='kernel')
        self.built = True

    @tf.function
    def call(self, inputs):
        weights = tf.nn.l2_normalize(self.kernel, axis=0)
        return tf.matmul(inputs, weights)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units,
                       "kernel_regularizer": self.kernel_regularizer})
        return config


def resnet101(input_shape, output_size, trainable=False, training=False,
              kernel_regularizer=None, name="resnetv2"):
    """This function returns a keras model of ResNet101.

    Args:
        input_shape: the shape of the inputs.
        output_size: size of output nodes. This is considered as the size of the 
            face embeddings.
        trainable: True if the model is open for traning.

    Returns:
        a keras model.
    """
    base_model = keras.applications.ResNet101V2(
        weights=None,
        input_shape=(112, 112, 3),
        pooling=None,
        include_top=False,)  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = trainable

    # Describe the model.
    inputs = keras.Input(input_shape, dtype=tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    x = base_model(x, training=training)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(output_size)(x)
    outputs = keras.layers.BatchNormalization()(x)

    # Construct the model and return it.
    model = keras.Model(inputs=inputs, outputs=outputs,
                        name=name, trainable=trainable)

    return model


if __name__ == "__main__":
    net = resnet((112, 112, 3), 256)
    x = tf.random.uniform((8, 112, 112, 3))
    x = net(x)
    print(x.shape)
