"""This module provides the network backbone implementation."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

from models.hrnet import hrnet_body


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

    def __init__(self, units, **kwargs):
        super(ArcLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-1], self.units],
                                      dtype=tf.float32,
                                      initializer=keras.initializers.HeNormal(),
                                      trainable=True,
                                      name='kernel')
        self.built = True

    @tf.function
    def call(self, inputs):
        weights = tf.nn.l2_normalize(self.kernel, axis=0)
        return tf.matmul(inputs, weights)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def hrnet_stem(filters=64):
    """The stem part of the network."""
    stem_layers = [layers.Conv2D(filters, 3, 2, 'same'),
                   layers.BatchNormalization(),
                   layers.Activation('relu')]

    def forward(x):
        for layer in stem_layers:
            x = layer(x)
        return x

    return forward


def hrnet_heads(input_channels=56, output_size=256):
    # Construct up sacling layers.
    scales = [2, 4, 8]
    up_scale_layers = [layers.UpSampling2D((s, s)) for s in scales]
    concatenate_layer = layers.Concatenate(axis=3)
    heads_layers = [layers.Conv2D(filters=input_channels, kernel_size=(1, 1),
                                  strides=(1, 1), padding='same'),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(output_size),
                    layers.BatchNormalization()]

    def forward(inputs):
        scaled = [f(x) for f, x in zip(up_scale_layers, inputs[1:])]
        x = concatenate_layer([inputs[0], scaled[0], scaled[1], scaled[2]])
        for layer in heads_layers:
            x = layer(x)
        return x

    return forward


def hrnet_v2(input_shape, output_size, width=18, trainable=True, name="hrnetv2"):
    """This function returns a keras model of HRNetV2.

    Args:
        input_shape: the shape of the inputs.
        output_size: size of output nodes. This is considered as the size of the 
            face embeddings.
        width: the model hyperparameter width.
        trainable: True if the model is open for traning.

    Returns:
        a keras model.
    """
    # Get the output size of the HRNet body.
    last_stage_width = sum([width * pow(2, n) for n in range(4)])

    # Describe the model.
    inputs = keras.Input(input_shape, dtype=tf.float32)
    x = hrnet_stem(64)(inputs)
    x = hrnet_body(width)(x)
    outputs = hrnet_heads(input_channels=last_stage_width,
                          output_size=output_size)(x)

    # Construct the model and return it.
    model = keras.Model(inputs=inputs, outputs=outputs,
                        name=name, trainable=trainable)

    return model


if __name__ == "__main__":
    net = hrnet_v2((112, 112, 3), 256)
    x = tf.random.uniform((8, 112, 112, 3))
    x = net(x)
    print(x.shape)
