import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLO, обученной на кубиках (замени "best.pt" на свою модель)
model = YOLO("yolov8n.pt")

# Открытие видеофайла
cap = cv2.VideoCapture("robotCam_1.avi")

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео")
    exit()

# Функция для определения цвета объекта
def get_cube_color(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]  # Вырезаем область кубика
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # Переводим в HSV
    
    # Определяем средний цвет (медиана)
    h, s, v = np.median(hsv[:, :, 0]), np.median(hsv[:, :, 1]), np.median(hsv[:, :, 2])

    # Определяем цвет по диапазонам HSV
    if (0 <= h <= 10) or (170 <= h <= 180):
        return "Red"
    elif 10 < h <= 25:
        return "Orange"
    elif 25 < h <= 35:
        return "Yellow"
    elif 35 < h <= 85:
        return "Green"
    elif 85 < h <= 130:
        return "Blue"
    else:
        return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Запускаем YOLO для детекции
    results = model(frame)

    # Обрабатываем найденные объекты
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты кубика
            confidence = box.conf[0].item()  # Уверенность
            class_id = int(box.cls[0])  # Класс объекта
            label = result.names[class_id]  # Имя класса (должно быть "cube")

            # Определяем цвет кубика
            cube_color = get_cube_color(frame, x1, y1, x2, y2)

            # Рисуем рамку и подпись с цветом
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cube_color} Cube {confidence:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Вычисляем центр кубика
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            print(f"Обнаружен {cube_color} кубик в ({cx}, {cy})")

    # Отображаем кадр
    cv2.imshow("YOLOv8 Cube Detection", frame)

    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
