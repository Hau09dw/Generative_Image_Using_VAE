import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE  # Import VAE từ thư mục models
from utils.data_loader1 import DataLoader  # Import DataLoader từ utils

class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.vae = VAE(
            latent_dim=config['latent_dim'], 
            input_shape=config['input_shape']
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=config['learning_rate'],
                decay_steps=10000,
                decay_rate=0.96,
                staircase=True
            )
        )
        log_dir = f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def compute_loss(self, inputs, reconstructed, z_mean, z_log_var, beta=1.0):
        # Sử dụng Binary Cross-Entropy (BCE) thay vì MSE
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        reconstruction_loss = tf.reduce_mean(bce(inputs, reconstructed))

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        kl_loss *= beta
        return reconstruction_loss + kl_loss

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var = self.vae(inputs)
            loss = self.compute_loss(inputs, reconstructed, z_mean, z_log_var, beta=1.0)
        gradients = tape.gradient(loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))
        return loss

    @tf.function
    def validate_step(self, inputs):
        reconstructed, z_mean, z_log_var = self.vae(inputs)
        loss = self.compute_loss(inputs, reconstructed, z_mean, z_log_var, beta=1.0)
        return loss

    def train(self, train_dataset, test_dataset):
        best_loss = float('inf')
        best_checkpoint_path = "results/checkpoints/vae_best.weights.h5"
        os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
        train_loss_history = []
        val_loss_history = []

        for epoch in range(self.config['epochs']):
            total_train_loss = 0
            batch_count = 0
            for batch in train_dataset:
                loss = self.train_step(batch)
                total_train_loss += loss
                batch_count += 1

            avg_train_loss = total_train_loss / batch_count
            train_loss_history.append(avg_train_loss.numpy())

            # Tính toán validation loss
            total_val_loss = 0
            val_batch_count = 0
            for batch in test_dataset:
                val_loss = self.validate_step(batch)
                total_val_loss += val_loss
                val_batch_count += 1

            avg_val_loss = total_val_loss / val_batch_count
            val_loss_history.append(avg_val_loss.numpy())

            with self.summary_writer.as_default():
                tf.summary.scalar('Training Loss', avg_train_loss, step=epoch)
                tf.summary.scalar('Validation Loss', avg_val_loss, step=epoch)

            print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss.numpy()}, Validation Loss: {avg_val_loss.numpy()}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                self.vae.save_weights(best_checkpoint_path)
                print(f"New best model saved at epoch {epoch+1} with validation loss {best_loss.numpy()}")

            if (epoch + 1) % 10 == 0:
                self.visualize_reconstruction(test_dataset, epoch)

        plot_training_loss(train_loss_history, val_loss_history)

    def visualize_reconstruction(self, dataset, epoch, save_dir='results/reconstructions'):
        os.makedirs(save_dir, exist_ok=True)
        for batch in dataset.take(1):
            reconstructed, _, _ = self.vae(batch)
            reconstructed = tf.clip_by_value(reconstructed, 0.0, 1.0)

            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for i in range(5):
                axes[0, i].imshow(batch[i].numpy())
                axes[0, i].set_title("Original")
                axes[0, i].axis('off')

                axes[1, i].imshow(reconstructed[i].numpy())
                axes[1, i].set_title("Reconstructed")
                axes[1, i].axis('off')

            plt.suptitle(f"Reconstruction at Epoch {epoch+1}")
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'reconstruction_epoch_{epoch+1}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved reconstruction image at: {save_path}")


def plot_training_loss(train_losses, val_losses, save_dir='results/plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_validation_loss_100e_raw2.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training and validation loss plot at: {save_path}")


def main():
    """
    Chương trình chính để khởi chạy huấn luyện VAE.
    """
    # Kiểm tra GPU
    if tf.test.is_gpu_available():
        print(f"GPU detected: {tf.test.gpu_device_name()}")
    else:
        print("No GPU detected. Please enable GPU in Colab.")

    # Cấu hình
    config = {
        'latent_dim': 1024,  # Tăng latent_dim để cải thiện chất lượng tái tạo
        'input_shape': (128, 128, 3),  # 3 kênh RGB
        'learning_rate': 1e-3,
        'epochs': 100,
        'batch_size': 32
    }

    # Tạo DataLoader
    data_dir = "/content/drive/MyDrive/Generative_Image_Using_VAE/data/raw2/img_align_celeba"
    data_loader = DataLoader(data_dir=data_dir, batch_size=config['batch_size'], img_size=config['input_shape'][:2])
    train_dataset, test_dataset = data_loader.load_dataset()

    # Kiểm tra kích thước batch (đảm bảo đúng định dạng)
    for batch in train_dataset.take(1):
        print(f"Sample batch shape: {batch.shape}")

    # Huấn luyện
    trainer = VAETrainer(config)
    trainer.train(train_dataset, test_dataset)


if __name__ == "__main__":
    main()
