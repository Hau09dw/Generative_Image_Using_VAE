import tensorflow as tf
import os

class DataLoader:
    def __init__(self, data_dir, img_size=(128,128), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
    
    def _preprocess_image(self, image_path):
        # Đọc và chuẩn hóa ảnh
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = image / 255.0  # Chuẩn hóa về [0,1]
        return image
    
    def load_dataset(self):
        # Tìm tất cả ảnh
        image_paths = tf.io.gfile.glob(os.path.join(self.data_dir, '*/*.jpg'))
        
        # Tạo dataset
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset