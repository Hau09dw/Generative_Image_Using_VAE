import tensorflow as tf
import wandb
from models.VAE import VAE
from utils.data_loader import DataLoader
import sys
import os
# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.vae = VAE(
            latent_dim=config['latent_dim'], 
            input_shape=config['input_shape']
        )
        
        self.optimizer = tf.keras.optimizers.Adam(config['learning_rate'])
        
    def compute_loss(self, inputs, reconstructed, z_mean, z_log_var):
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
    
    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var = self.vae(inputs)
            loss = self.compute_loss(inputs, reconstructed, z_mean, z_log_var)
        
        gradients = tape.gradient(loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))
        
        return loss
    
    def train(self, dataset):
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for batch in dataset:
                loss = self.train_step(batch)
                total_loss += loss
            
            print(f"Epoch {epoch+1}, Loss: {total_loss.numpy()}")
            
            # Lưu checkpoint
            self.vae.save_weights(f"results/checkpoints/vae_epoch_{epoch+1}")

def main():
    # Cấu hình
    config = {
        'latent_dim': 200,
        'input_shape': (128, 128, 3),
        'learning_rate': 1e-4,
        'epochs': 100,
        'batch_size': 32
    }
    
    # Tải dữ liệu
    data_loader = DataLoader('./data/processed', batch_size=config['batch_size'])
    dataset = data_loader.load_dataset()
    
    # Huấn luyện
    trainer = VAETrainer(config)
    trainer.train(dataset)

if __name__ == "__main__":
    main()