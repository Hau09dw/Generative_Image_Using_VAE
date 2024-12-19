import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.VAE import VAE  # Import VAE sau khi thêm đường dẫn đúng


def generate_images(vae, num_images=16):
    """
    Sinh ảnh từ không gian tiềm năng và lưu trữ.
    
    Args:
        vae (VAE): Mô hình VAE đã tải trọng số.
        num_images (int): Số lượng ảnh cần sinh.
    """
    # Tạo vector ngẫu nhiên trong không gian tiềm năng
    z_sample = np.random.normal(0, 1, (num_images, vae.latent_dim))
    
    # Giải mã từ không gian tiềm năng
    generated_images = vae.decoder.predict(z_sample)
    generated_images = np.clip(generated_images, 0, 1)  # Đảm bảo ảnh nằm trong khoảng [0, 1]
    
    # Hiển thị và lưu ảnh
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray' if generated_images[i].shape[-1] == 1 else None)
        plt.axis('off')
    
    # Đảm bảo thư mục tồn tại
    save_dir = 'results/generated_images/'
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generated.png'))
    plt.show()


def main():
    """
    Tải mô hình VAE và sinh ảnh từ không gian tiềm năng.
    """
    # Kiểm tra GPU
    if tf.test.is_gpu_available():
        print("GPU detected. Using GPU for inference.")
    else:
        print("No GPU detected. Running on CPU.")

    # Tải mô hình
    latent_dim = 256  # Đảm bảo latent_dim giống với cấu hình huấn luyện
    input_shape = (128, 128, 3)
    checkpoint_path = '/content/Generative_Image_Using_VAE/results/checkpoints/vae_epoch_20.weights.h5'  # Đường dẫn đến checkpoint

    # Kiểm tra checkpoint tồn tại
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} không tồn tại!")

    vae = VAE(latent_dim=latent_dim, input_shape=input_shape)
    try:
        vae.load_weights(checkpoint_path)
        print(f"Checkpoint {checkpoint_path} đã được tải thành công.")
    except Exception as e:
        print(f"Không thể tải checkpoint: {e}")
        return

    # Sinh ảnh
    generate_images(vae, num_images=16)


if __name__ == "__main__":
    main()
