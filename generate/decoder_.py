from tensorflow.keras import layers
import tensorflow as tf


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self,
                 original_dim,
                 filters=32,
                 kernel_size=3,
                 activation=tf.nn.leaky_relu,
                 strides=1,
                 padding="same",
                 dropout_rate=0.2,
                 use_batchnorm=True,
                 name="decoder",
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.original_dim = original_dim

    def build(self, input_shape):
        self.deconv1 = tf.keras.layers.Conv1DTranspose(input_shape=input_shape,
                                                       filters=self.filters,
                                                       kernel_size=self.kernel_size,
                                                       strides=self.strides,
                                                       padding=self.padding,
                                                       activation=self.activation)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.deconv2 = tf.keras.layers.Conv1DTranspose(filters=self.filters,
                                                       kernel_size=self.kernel_size,
                                                       strides=self.strides,
                                                       padding=self.padding,
                                                       activation=self.activation)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.deconv3 = tf.keras.layers.Conv1DTranspose(filters=self.filters,
                                                       kernel_size=self.kernel_size,
                                                       strides=self.strides,
                                                       padding=self.padding,
                                                       activation=self.activation)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense_output = tf.keras.layers.Dense(self.original_dim,
                                                  activation="tanh")

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)

        x = self.deconv1(x)
        x = self.bn1(x)

        x = self.deconv2(x)
        x = self.bn2(x)

        x = self.deconv3(x)
        x = self.bn3(x)

        x = self.dropout(x)

        x = self.flatten(x)

        return self.dense_output(x)

# def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
#     super(Decoder, self).__init__(name=name, **kwargs)
#     self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
#     self.dense_output = layers.Dense(original_dim, activation="sigmoid")
#
# def call(self, inputs):
#     x = self.dense_proj(inputs)
#     return self.dense_output(x)
