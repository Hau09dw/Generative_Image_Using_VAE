import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.VAE import VAE

def generate_images(vae, num_images=16, latent_dim=200):
    # Sinh ngẫu nhiên trong không gian tiềm năng
    z_sample = np.random.normal(0, 1, (num_images, latent_dim))
    
    # Giải mã
    generated_images = vae.decoder.predict(z_sample)
    
    # Hiển thị
    plt.figure(figsize=(10,10))
    for i in range(num_images):
        plt.subplot(4,4,i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/generated_images/generated_faces.png')

def main():
    # Tải mô hình
    vae = VAE(latent_dim=200, input_shape=(128,128,3))
    vae.load_weights('results/checkpoints/latest')
    
    # Sinh ảnh
    generate_images(vae)

if __name__ == "__main__":
    main()  