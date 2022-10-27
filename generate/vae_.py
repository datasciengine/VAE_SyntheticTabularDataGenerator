import tensorflow as tf
import keras
from generate.encoder_ import Encoder
from generate.decoder_ import Decoder


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim,
                 latent_dim=32,
                 name="autoencoder",
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(original_dim)

    def call(self, inputs):
        """
        :param inputs:
        :return:
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)  # custom loss ekliyor mse += kl_loss
        return reconstructed
