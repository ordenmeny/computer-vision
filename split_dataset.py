import os
import shutil
import random

# Пути к файлам
dataset_path = "dataset_images/all_dataset"
image_dir = os.path.join(dataset_path, "images")
label_dir = os.path.join(dataset_path, "labels")

# Создаём новые папки
split_dirs = ["train", "val", "test"]
for split in split_dirs:
    os.makedirs(f"dataset_images/all_dataset/{split}/images", exist_ok=True)
    os.makedirs(f"dataset_images/all_dataset/{split}/labels", exist_ok=True)

# Получаем список файлов
images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
random.shuffle(images)

# Определяем пропорции
train_split = int(len(images) * 0.8)  # 80% для train
val_split = int(len(images) * 0.9)    # 10% для val, 10% для test

for i, img in enumerate(images):
    label = img.replace(".jpg", ".txt")
    
    if i < train_split:
        split = "train"
    elif i < val_split:
        split = "val"
    else:
        split = "test"

    shutil.move(os.path.join(image_dir, img), f"dataset_images/all_dataset/{split}/images/{img}")
    shutil.move(os.path.join(label_dir, label), f"dataset_images/all_dataset/{split}/labels/{label}")

print("Данные разделены на train/val/test!")
