import tensorflow as tf

class DataLoader:
    def __init__(self, batch_size=32, img_size=(28, 28), sample_limit=20000):
        """
        DataLoader để tải và xử lý dữ liệu MNIST với số lượng mẫu giới hạn.

        Args:
            batch_size (int): Kích thước batch.
            img_size (tuple): Kích thước đầu vào (H, W).
            sample_limit (int): Số lượng mẫu giới hạn để sử dụng.
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.sample_limit = sample_limit

    def _preprocess_image(self, image):
        """
        Xử lý ảnh (resize, chuyển grayscale sang RGB, chuẩn hóa).

        Args:
            image (tf.Tensor): Ảnh đầu vào.

        Returns:
            tf.Tensor: Ảnh được resize và chuẩn hóa.
        """
        # Chuyển đổi sang kiểu float32
        image = tf.cast(image, tf.float32)
        # Chuyển từ grayscale (1 kênh) sang RGB (3 kênh)
        image = tf.image.grayscale_to_rgb(image)
        # Resize ảnh
        image = tf.image.resize(image, self.img_size)
        # Chuẩn hóa ảnh về khoảng [0, 1]
        image = image / 255.0
        return image

    def load_dataset(self, digit=None):
        """
        Tải và xử lý dataset MNIST với số lượng mẫu giới hạn và có thể lọc theo số cụ thể.

        Args:
            digit (int, optional): Số cụ thể muốn lọc (0-9). Nếu None, sử dụng tất cả các số.

        Returns:
            tuple: (train_dataset, test_dataset)
        """
        # Tải dataset MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Lọc dữ liệu theo số cụ thể nếu được cung cấp
        if digit is not None:
            train_filter = y_train == digit
            test_filter = y_test == digit
            x_train, y_train = x_train[train_filter], y_train[train_filter]
            x_test, y_test = x_test[test_filter], y_test[test_filter]

        # Giới hạn số lượng mẫu
        x_train = x_train[:self.sample_limit]  # Lấy tối đa sample_limit mẫu từ tập train
        x_test = x_test[:self.sample_limit]    # Lấy tối đa sample_limit mẫu từ tập test

        # Thêm trục kênh vào dữ liệu ảnh
        x_train = x_train[..., tf.newaxis]  # (batch_size, height, width, 1)
        x_test = x_test[..., tf.newaxis]    # (batch_size, height, width, 1)

        # Tạo dataset từ tensor
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

        # Xử lý dữ liệu
        train_dataset = (train_dataset
                         .map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                         .shuffle(1000)
                         .batch(self.batch_size)
                         .prefetch(tf.data.AUTOTUNE))

        test_dataset = (test_dataset
                        .map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(self.batch_size)
                        .prefetch(tf.data.AUTOTUNE))

        return train_dataset, test_dataset
