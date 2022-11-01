import tensorflow as tf
from generate.sampling_ import Sampling


class Encoder(tf.keras.layers.Layer):

    def __init__(self,
                 latent_dim=32,
                 filters=32,
                 kernel_size=3,
                 activation=tf.nn.leaky_relu,
                 strides=1,
                 padding="same",
                 dropout_rate=0.2,
                 use_batchnorm=True,
                 name="encoder",
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.latent_dim = latent_dim

        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv1D(input_shape=input_shape,
                                            filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            activation=self.activation)

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv1D(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            activation=self.activation)

        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv1D(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            activation=self.activation)

        self.bn3 = tf.keras.layers.BatchNormalization()

        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.flatten = tf.keras.layers.Flatten()

        self.dense_mean = tf.keras.layers.Dense(self.latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(self.latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        # input was 128, 33
        x = tf.expand_dims(inputs, axis=-1)
        # now its 128, 33, 1

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.dropout(x)

        x = self.flatten(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
