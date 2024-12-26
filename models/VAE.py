import tensorflow as tf
import numpy as np

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, input_shape=(128, 128, 3), latent_dim=512, name='vae', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder(input_shape, latent_dim)
        self.decoder = self._build_decoder(input_shape, latent_dim)

    def _build_encoder(self, input_shape, latent_dim):
        inputs = tf.keras.Input(shape=input_shape)

        x = tf.keras.layers.Conv2D(32, 4, strides=2, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SpatialDropout2D(0.2)(x)

        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SpatialDropout2D(0.3)(x)

        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

        z = Sampling()([z_mean, z_log_var])

        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def _build_decoder(self, input_shape, latent_dim):
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        
        x = tf.keras.layers.Dense(16 * 16 * 128, activation='relu')(latent_inputs)
        x = tf.keras.layers.Reshape((16, 16, 128))(x)

        x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        outputs = tf.keras.layers.Conv2DTranspose(
            input_shape[2], 3, padding='same', activation='sigmoid'
        )(x)

        return tf.keras.Model(latent_inputs, outputs, name='decoder')

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var
