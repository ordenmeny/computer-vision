import cv2
import numpy as np

# Загружаем видео
cap = cv2.VideoCapture("robotCam_1.avi")

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео")
    exit()

# Функция для создания маски по цвету (HSV)
def color_mask(hsv, lower, upper):
    return cv2.inRange(hsv, np.array(lower), np.array(upper))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Диапазоны HSV для каждого цвета кубика
    lower_red1, upper_red1 = [0, 120, 70], [10, 255, 255]  # Красный (оттенок 1)
    lower_red2, upper_red2 = [170, 120, 70], [180, 255, 255]  # Красный (оттенок 2)
    lower_orange, upper_orange = [10, 100, 100], [25, 255, 255]  # Оранжевый
    lower_yellow, upper_yellow = [25, 100, 100], [35, 255, 255]  # Желтый
    lower_green, upper_green = [35, 50, 50], [85, 255, 255]  # Зеленый
    lower_blue, upper_blue = [85, 50, 50], [130, 255, 255]  # Синий

    # Создаем маски для каждого цвета
    mask_red = color_mask(hsv, lower_red1, upper_red1) + color_mask(hsv, lower_red2, upper_red2)
    mask_orange = color_mask(hsv, lower_orange, upper_orange)
    mask_yellow = color_mask(hsv, lower_yellow, upper_yellow)
    mask_green = color_mask(hsv, lower_green, upper_green)
    mask_blue = color_mask(hsv, lower_blue, upper_blue)

    # Объединяем маски
    mask = mask_red + mask_orange + mask_yellow + mask_green + mask_blue

    # Убираем шумы с помощью морфологических операций
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000:  # Фильтр по размеру
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:  # Если 4 угла (похоже на квадрат)
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

                # Найдем центр контура
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Подписываем найденный кубик
                    cv2.putText(frame, f"Cube {i} ({cx}, {cy})", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Показываем результат
    cv2.imshow("Cubes Detection", frame)

    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
