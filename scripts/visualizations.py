import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_metrics(metrics, save_dir='results/plots'):
    """
    Vẽ biểu đồ Loss (Training Loss, Validation Loss) và KL Divergence.

    Args:
        metrics (dict): Dictionary chứa các giá trị loss, kl_divergence, reconstruction_loss.
        save_dir (str): Đường dẫn lưu các biểu đồ.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['training_loss'], label='Training Loss')
    plt.plot(metrics['validation_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    # KL Divergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['kl_divergence'], label='KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'kl_divergence_plot.png'))
    plt.close()

    # Reconstruction Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['reconstruction_loss'], label='Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'reconstruction_loss_plot.png'))
    plt.close()


def plot_latent_space(vae, data, labels, save_path='results/plots/latent_space.png'):
    """
    Vẽ không gian tiềm năng (Latent Space Visualization).

    Args:
        vae: Mô hình VAE.
        data (numpy.ndarray): Dữ liệu đầu vào.
        labels (numpy.ndarray): Nhãn tương ứng.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def compare_reconstructed_images(original, reconstructed, save_path='results/plots/reconstructed_vs_original.png'):
    """
    So sánh ảnh gốc và ảnh tái tạo.

    Args:
        original (numpy.ndarray): Ảnh gốc.
        reconstructed (numpy.ndarray): Ảnh tái tạo từ mô hình.
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    n = min(10, len(original))  # Số lượng ảnh so sánh
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Ảnh gốc
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray' if original.shape[-1] == 1 else None)
        plt.title("Original")
        plt.axis("off")

        # Ảnh tái tạo
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray' if reconstructed.shape[-1] == 1 else None)
        plt.title("Reconstructed")
        plt.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_learning_rate_schedule(schedule, steps, save_path='results/plots/learning_rate_schedule.png'):
    """
    Vẽ biểu đồ Learning Rate Schedule.

    Args:
        schedule: Biểu đồ learning rate.
        steps (int): Tổng số bước (iterations).
        save_path (str): Đường dẫn lưu biểu đồ.
    """
    lrs = [schedule(step) for step in range(steps)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(steps), lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
