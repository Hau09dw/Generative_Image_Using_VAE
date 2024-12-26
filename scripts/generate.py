import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.VAE import VAE

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_gpu():
    if tf.config.list_physical_devices('GPU'):
        logging.info("GPU detected. Using GPU for inference.")
    else:
        logging.warning("No GPU detected. Running on CPU.")

def generate_faces(vae, n_to_show=30, save_dir='results/generated_faces'):
    os.makedirs(save_dir, exist_ok=True)
    znew = np.random.normal(size=(n_to_show, vae.latent_dim))
    reconst = vae.decoder.predict(znew, batch_size=16)

    fig = plt.figure(figsize=(18, 5))
    for i in range(n_to_show):
        ax = fig.add_subplot(3, 10, i + 1)
        ax.imshow(reconst[i])
        ax.axis('off')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join(save_dir, f'generated_faces_{timestamp}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Generated faces saved at: {save_path}")

def interpolate_faces(vae, n_steps=10, save_dir='results/interpolated_faces'):
    os.makedirs(save_dir, exist_ok=True)
    z1, z2 = np.random.normal(size=(1, vae.latent_dim)), np.random.normal(size=(1, vae.latent_dim))
    alphas = np.linspace(0, 1, n_steps)
    z_interp = np.array([z1 * (1 - alpha) + z2 * alpha for alpha in alphas]).reshape(-1, vae.latent_dim)
    reconst = vae.decoder.predict(z_interp, batch_size=16)

    fig = plt.figure(figsize=(20, 4))
    for i in range(n_steps):
        ax = fig.add_subplot(1, n_steps, i + 1)
        ax.imshow(reconst[i])
        ax.axis('off')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join(save_dir, f'interpolated_faces_{timestamp}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Interpolated faces saved at: {save_path}")

def main():
    check_gpu()
    
    # Cấu hình
    config = {
        'latent_dim': 512,
        'input_shape': (128, 128, 3),
        'checkpoint_path': 'results/checkpoints/vae_best.weights.h5'
    }

    if not os.path.exists(config['checkpoint_path']):
        logging.error(f"Checkpoint {config['checkpoint_path']} does not exist!")
        return

    vae = VAE(latent_dim=config['latent_dim'], input_shape=config['input_shape'])
    try:
        vae.load_weights(config['checkpoint_path'])
        logging.info(f"Checkpoint {config['checkpoint_path']} loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return

    logging.info("Generating new faces...")
    generate_faces(vae)

    logging.info("Creating interpolations between faces...")
    interpolate_faces(vae)

if __name__ == "__main__":
    main()
