import cv2
import torch
from ultralytics import YOLO

# Загружаем обученную модель YOLO (можно заменить на свою best.pt)
model = YOLO("yolov8n.pt")

# Открываем видеофайл
cap = cv2.VideoCapture("robotCam_1.avi")

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Запускаем YOLOv8 на кадре
    results = model(frame)

    # Обрабатываем результаты детекции
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты рамки
            confidence = box.conf[0].item()  # Уверенность модели
            class_id = int(box.cls[0])  # Класс объекта
            label = result.names[class_id]  # Имя класса (например, "cube")

            # Отрисовываем рамку вокруг объекта
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Выводим текст (название + уверенность)
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Вычисляем центр объекта
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            print(f"Обнаружен {label} ({cx}, {cy})")

    # Отображаем кадр
    cv2.imshow("YOLOv8 Cube Detection", frame)

    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()