import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class VAE(tf.keras.Model):
    def __init__(
        self, 
        input_shape=(128, 128, 3), 
        latent_dim=200, 
        name='vae', 
        **kwargs
    ):
        """
        Khởi tạo mô hình Variational Autoencoder
        
        Parameters:
        -----------
        input_shape : tuple
            Kích thước đầu vào của ảnh
        latent_dim : int
            Số chiều của không gian tiềm năng
        """
        super(VAE, self).__init__(name=name, **kwargs)
        
        # Encoder
        self.encoder = self._build_encoder(input_shape, latent_dim)
        
        # Decoder
        self.decoder = self._build_decoder(input_shape, latent_dim)
        
        # Thuộc tính lưu trữ
        self.input_shape = input_shape
        self.latent_dim = latent_dim
    
    def _build_encoder(self, input_shape, latent_dim):
        """Xây dựng mạng encoder"""
        inputs = tf.keras.Input(shape=input_shape)
        
        x = tf.keras.layers.Conv2D(32, 4, strides=2, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
        
        return tf.keras.Model(inputs, [z_mean, z_log_var], name='encoder')
    
    def _build_decoder(self, input_shape, latent_dim):
        """Xây dựng mạng decoder"""
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        
        x = tf.keras.layers.Dense(
            np.prod(input_shape[:-1]) // 8, 
            activation='relu'
        )(latent_inputs)
        
        x = tf.keras.layers.Reshape(
            (input_shape[0]//8, input_shape[1]//8, 128)
        )(x)
        
        x = tf.keras.layers.Conv2DTranspose(
            128, 4, strides=2, padding='same', activation='relu'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(
            64, 4, strides=2, padding='same', activation='relu'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(
            32, 4, strides=2, padding='same', activation='relu'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        outputs = tf.keras.layers.Conv2DTranspose(
            input_shape[2], 3, padding='same', activation='sigmoid'
        )(x)
        
        return tf.keras.Model(latent_inputs, outputs, name='decoder')
    
    def reparameterize(self, z_mean, z_log_var):
        """
        Reparameterization trick
        Sinh mẫu từ phân phối tiềm năng
        """
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def call(self, inputs):
        """
        Forward pass của mô hình
        """
        # Encode
        z_mean, z_log_var = self.encoder(inputs)
        
        # Reparameterize
        z = self.reparameterize(z_mean, z_log_var)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, z_mean, z_log_var
    
    def sample(self, num_samples=1):
        """
        Sinh ảnh từ không gian tiềm năng
        """
        z = np.random.normal(0, 1, (num_samples, self.latent_dim))
        return self.decoder.predict(z)
    
    def compute_loss(self, inputs, reconstructed, z_mean, z_log_var):
        """
        Tính toán loss function
        """
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(inputs, reconstructed), 
                axis=[1,2,3]
            )
        )
        
        # KL Divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        
        return reconstruction_loss + kl_loss