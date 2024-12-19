import tensorflow as tf
import numpy as np


class VAE(tf.keras.Model):
    def __init__(
        self,
        input_shape=(128, 128, 3),
        latent_dim=512,
        name='vae',
        **kwargs
    ):
        super(VAE, self).__init__(name=name, **kwargs)
        self.encoder = self._build_encoder(input_shape, latent_dim)
        self.decoder = self._build_decoder(input_shape, latent_dim)
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    def _build_encoder(self, input_shape, latent_dim):
        inputs = tf.keras.Input(shape=input_shape)

        x = tf.keras.layers.Conv2D(32, 4, strides=2, padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(256, activation='relu')(x)
        z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

        return tf.keras.Model(inputs, [z_mean, z_log_var], name='encoder')

    def _build_decoder(self, input_shape, latent_dim):
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(16 * 16 * 128, activation='relu')(latent_inputs)
        x = tf.keras.layers.Reshape((16, 16, 128))(x)

        x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        outputs = tf.keras.layers.Conv2DTranspose(
            input_shape[2], 3, padding='same', activation='sigmoid'
        )(x)

        return tf.keras.Model(latent_inputs, outputs, name='decoder')

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var
