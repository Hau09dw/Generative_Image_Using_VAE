import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Thư mục chứa dữ liệu
inp_dir = '../data/raw'
target_size = (128, 128)

# Đọc tất cả hình ảnh và nhãn
all_images = []

img_list = os.listdir(inp_dir)  # Lấy tất cả file trong thư mục raw
for img in img_list:
    fname = os.path.join(inp_dir, img)
    if os.path.isfile(fname):  # Kiểm tra nếu là file
        try:
            # Load và resize ảnh
            image = load_img(fname, target_size=target_size)
            image = img_to_array(image).astype('float32') / 255.0  # Chuẩn hóa
            all_images.append(image)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

# Chuyển đổi sang mảng numpy
all_images = np.array(all_images)

# Tách dữ liệu train/test
x_train, x_test = train_test_split(all_images, test_size=0.2, random_state=42)

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs('./data/processed', exist_ok=True)

# Lưu dữ liệu
np.save('../data/processed/x_train.npy', x_train)
np.save('../data/processed/x_test.npy', x_test)

print("Preprocessing hoàn thành! Dữ liệu đã được lưu.")
