import tensorflow as tf
import os
import glob

class CustomImageDataLoader:
    def init(self, data_dir, batch_size=32, img_size=(64, 64)):
        """
        DataLoader để tải và xử lý ảnh từ thư mục.

        Args:
            data_dir (str): Đường dẫn đến thư mục chứa ảnh.
            batch_size (int): Kích thước batch.
            img_size (tuple): Kích thước đầu vào (H, W).
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def _load_and_preprocess_image(self, image_path):
        """
        Đọc và xử lý ảnh từ đường dẫn.

        Args:
            image_path (str): Đường dẫn đến file ảnh.

        Returns:
            tf.Tensor: Ảnh đã được xử lý.
        """
        # Đọc file ảnh
        image = tf.io.read_file(image_path)
        # Giải mã JPG
        image = tf.image.decode_jpeg(image, channels=3)
        # Chuyển đổi sang kiểu float32
        image = tf.cast(image, tf.float32)
        # Resize ảnh
        image = tf.image.resize(image, self.img_size)
        # Chuẩn hóa ảnh về khoảng [0, 1]
        image = image / 255.0
        return image

    def load_dataset(self, validation_split=0.2):
        """
        Tải và xử lý dataset từ thư mục ảnh.

        Args:
            validation_split (float): Tỷ lệ chia tập validation (0.0 đến 1.0).

        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # Lấy danh sách đường dẫn ảnh
        image_paths = glob.glob(os.path.join(self.data_dir, "*.jpg"))
        total_images = len(image_paths)

        if total_images == 0:
            raise ValueError(f"Không tìm thấy ảnh JPG trong thư mục {self.data_dir}")

        # Tạo dataset từ đường dẫn ảnh
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        # Xáo trộn dataset
        dataset = dataset.shuffle(buffer_size=total_images)

        # Chia dataset
        val_size = int(total_images * validation_split)
        train_size = total_images - val_size

        # Tạo tập train và validation
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        # Xử lý dữ liệu cho tập train
        train_dataset = (train_dataset
                        .map(self._load_and_preprocess_image, 
                             num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(self.batch_size)
                        .prefetch(tf.data.AUTOTUNE))

        # Xử lý dữ liệu cho tập validation
        val_dataset = (val_dataset
                      .map(self._load_and_preprocess_image,
                           num_parallel_calls=tf.data.AUTOTUNE)
                      .batch(self.batch_size)
                      .prefetch(tf.data.AUTOTUNE))

        return train_dataset, val_dataset