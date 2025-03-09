import os
import cv2
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Папка с исходными изображениями
input_dir = "dataset_images/original"
# Папка для сохранения аугментированных изображений
output_dir = "dataset_images/augmented"

os.makedirs(output_dir, exist_ok=True)

# Определяем аугментации
transform = A.Compose([
    A.Rotate(limit=30, p=0.7),  # Поворот на ±30 градусов
    A.HorizontalFlip(p=0.5),  # Отражение
    A.VerticalFlip(p=0.2),  # Вертикальное отражение
    A.RandomBrightnessContrast(p=0.5),  # Яркость и контраст
    A.GaussNoise(std_range=(0.05, 0.15), p=1.0),  # Добавление шума
    A.MotionBlur(blur_limit=5, p=0.3),  # Размытие
    ToTensorV2()
])

# Обрабатываем изображения
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(5):  # Создаем 5 аугментированных версий каждого изображения
        augmented = transform(image=img)['image']
        augmented_np = augmented.permute(1, 2, 0).numpy()  # Переводим обратно в OpenCV формат
        save_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_aug_{i}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR))

print("Аугментация завершена!")
