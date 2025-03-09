from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")  # Загружаем предобученную модель YOLOv8
    # Запускаем обучение
    model.train(
        data=r"F:\ComputerMagic\dataset_images\all_dataset\cubes.yaml",  # Указываем путь к конфигу датасета
        epochs=50,  # Количество эпох (чем больше, тем лучше)
        imgsz=640,  # Размер входного изображения
        batch=16,  # Размер batch (сколько картинок сразу обрабатывается)
        device="cuda"
    )
