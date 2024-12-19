import tensorflow as tf
import os
import sys
from datetime import datetime

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE  # Import VAE từ thư mục models
from utils.data_loader1 import CustomImageDataLoader as DataLoader  # Import DataLoader từ utils


class VAETrainer:
    def __init__(self, config):
        """
        Khởi tạo lớp huấn luyện VAE.
        
        Args:
            config (dict): Các thông số cấu hình như latent_dim, input_shape, learning_rate.
        """
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
        # Thiết lập TensorBoard
        log_dir = f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        
    # def compute_loss(self, inputs, reconstructed, z_mean, z_log_var):
    #     """
    #     Tính toán hàm mất mát bao gồm loss tái tạo và KL Divergence.
    #     """
    #     # Reconstruction loss (sử dụng MeanSquaredError từ tf.keras.losses)
    #     mse = tf.keras.losses.MeanSquaredError()
    #     reconstruction_loss = tf.reduce_mean(mse(inputs, reconstructed))
        
    #     # KL Divergence loss
    #     kl_loss = -0.5 * tf.reduce_mean(
    #         1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    #     )
        
    #     return reconstruction_loss + kl_loss
    
    # @tf.function
    # def train_step(self, inputs):
    #     """
    #     Một bước huấn luyện mô hình.
    #     """
    #     with tf.GradientTape() as tape:
    #         reconstructed, z_mean, z_log_var = self.vae(inputs)
    #         loss = self.compute_loss(inputs, reconstructed, z_mean, z_log_var)
        
    #     gradients = tape.gradient(loss, self.vae.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))
        
    #     return loss

    # #Beta VAE
    def compute_loss(self, inputs, reconstructed, z_mean, z_log_var, beta=4.0):
        """
        Tính toán hàm mất mát Beta-VAE bao gồm reconstruction loss và beta * KL divergence.

        Args:
            inputs (tf.Tensor): Dữ liệu đầu vào.
            reconstructed (tf.Tensor): Dữ liệu tái tạo từ mô hình.
            z_mean (tf.Tensor): Giá trị trung bình không gian ẩn.
            z_log_var (tf.Tensor): Log phương sai không gian ẩn.
            beta (float): Hệ số điều chỉnh mức độ ưu tiên của KL divergence.

        Returns:
            tf.Tensor: Hàm mất mát tổng hợp.
        """
        # Reconstruction loss
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = tf.reduce_mean(mse(inputs, reconstructed))
        
        # KL Divergence loss với hệ số beta
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        kl_loss *= beta  # Thêm hệ số beta
        
        return reconstruction_loss + kl_loss

    @tf.function
    def train_step(self, inputs):
        """
        Một bước huấn luyện mô hình Beta-VAE.
        """
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var = self.vae(inputs)
            loss = self.compute_loss(inputs, reconstructed, z_mean, z_log_var, beta=4.0)  # Thay đổi beta tại đây
        
        gradients = tape.gradient(loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))
        
        return loss



    
    def train(self, dataset):
        """
        Huấn luyện mô hình trên tập dữ liệu.
        """
        for epoch in range(self.config['epochs']):
            total_loss = 0
            batch_count = 0
            for batch in dataset:
                loss = self.train_step(batch)
                total_loss += loss
                batch_count += 1
            
            avg_loss = total_loss / batch_count

            # Log thông tin lên TensorBoard
            with self.summary_writer.as_default():
                tf.summary.scalar('Loss', avg_loss, step=epoch)

            print(f"Epoch {epoch+1}, Loss: {avg_loss.numpy()}")

            # Lưu checkpoint với định dạng .weights.h5
            if (epoch + 1) % 20 == 0:  # Save every 10 epochs
                checkpoint_path = f"results/checkpoints/vae_epoch_{epoch+1}.weights.h5"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                self.vae.save_weights(checkpoint_path)

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
        'latent_dim': 516,  # Đảm bảo latent_dim giống với cấu hình huấn luyện
        'input_shape': (128, 128, 3),  # 3 kênh RGB
        'learning_rate': 1e-3,
        'epochs': 100,
        'batch_size': 32
    }
    """config = {
        'latent_dim': 512,  # Đảm bảo latent_dim giống với cấu hình huấn luyện
        'input_shape': (128, 128, 3),  # 3 kênh RGB
        'learning_rate': 1e-3,
        'epochs': 50,
        'batch_size': 32
    }
"""
    # Tạo DataLoader
    #data_loader = DataLoader(batch_size=config['batch_size'], img_size=config['input_shape'][:2])
    data_dir = "/content/Generative_Image_Using_VAE/data/raw"
    data_loader = DataLoader(data_dir=data_dir,batch_size=config['batch_size'], img_size=config['input_shape'][:2])

    # Tải dữ liệu chỉ chứa số cụ thể, ví dụ số 5
    #train_dataset, test_dataset = data_loader.load_dataset(digit=5)
    train_dataset, test_dataset = data_loader.load_dataset()

    # Kiểm tra kích thước batch (đảm bảo đúng định dạng)
    for batch in train_dataset.take(1):
        print(f"Sample batch shape: {batch.shape}")

    # Huấn luyện
    trainer = VAETrainer(config)
    trainer.train(train_dataset)


if __name__ == "__main__":
    main()