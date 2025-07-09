import math
import mediapipe as mp
import cv2
from pythonosc import udp_client
import numpy as np
from collections import deque

# Налаштування OSC
IP = "127.0.0.1"
PORT = 9000

# Налаштування OSC-клієнта
client = udp_client.SimpleUDPClient(IP, PORT)

# Параметри згладжування
SMOOTHING_FACTOR = 0.5  # Коефіцієнт експонентного згладжування (менше = більше згладжування)
HISTORY_SIZE = 5  # Розмір буфера для ковзного середнього
SMILE_COMPENSATION = 0.6  # Коэффициент компенсации влияния улыбки на открытие рта

# Параметри посилення посмішки
SMILE_POWER = 0.5  # Показник ступеня для посилення посмішки (менше = більше посилення)
SMILE_SCALE = 1.4  # Масштабуючий коефіцієнт для посмішки (більше = сильніша за посмішку)

# Параметры формы "О"
O_SHAPE_RATIO_THRESHOLD = 0.1  # Порогове значення для розпізнавання форми "О" (співвідношення висоти до ширини)
O_SHAPE_CORNERS_THRESHOLD = 0.1  # Максимальне значення підняття куточків рота для форми "О"
MAX_COMBINED_VALUE = 1.5  # Максимальне сумарне значення Mouth_O + Mouth_Open

# Параметри відстеження очей
EYE_SMOOTHING_FACTOR = 0.4  # Коефіцієнт експоненціального згладжування для очей
EYE_HISTORY_SIZE = 5  # Розмір буфера для ковзного середнього
EYE_SCALE = 1.3  # Масштабуючий коефіцієнт для руху очей (більше = більший діапазон руху)

# Параметри відстеження кліпання
BLINK_SMOOTHING_FACTOR = 0.3  # Коефіцієнт згладжування для кліпання
BLINK_HISTORY_SIZE = 3  # Розмір буфера для ковзного середнього кліпання
BLINK_THRESHOLD = 0.35  # Порогове значення для визначення кліпання


# Константи для нормалізації відкриття рота
MIN_MOUTH_DISTANCE = 20
MAX_MOUTH_DISTANCE = 60
MIN_MOUTH_WIDTH = 0  # зміниться після калібрування
MAX_MOUTH_WIDTH = 100  # зміниться після калібрування

# Константи для нормалізації руху куточків рота
MIN_LEFT_CORNER_DISTANCE = 0  # зміниться після калібрування
MAX_LEFT_CORNER_DISTANCE = 20  # зміниться після калібрування
MIN_RIGHT_CORNER_DISTANCE = 0  # зміниться після калібрування
MAX_RIGHT_CORNER_DISTANCE = 20  # зміниться після калібрування

# Параметри відстеження брів
BROW_SMOOTHING_FACTOR = 0.3  # Коефіцієнт згладжування для брів
BROW_HISTORY_SIZE = 4  # Розмір буфера для ковзного середнього
BROW_FROWN_SCALE = 1.3  # Масштабуючий коефіцієнт для хмуріння брів
BROW_RAISE_SCALE = 1.2  # Масштабуючий коефіцієнт для підняття брів



def send_mouth_openness(value):
    value = max(0, min(0.99, value))
    client.send_message("/avatar/parameters/Mouth_Open", value)
    print(f"Sent mouth openness: {value:.2f}")

def send_mouth_o_shape(value):
    value = max(0, min(0.99, value))
    client.send_message("/avatar/parameters/Mouth_O", value)
    print(f"Sent mouth O shape: {value:.2f}")

def send_left_corner(value):
    value = max(0, min(1, value))
    client.send_message("/avatar/parameters/Smile_L", value)
    print(f"Sent left corner: {value:.2f}")

def send_right_corner(value):
    value = max(0, min(1, value))
    client.send_message("/avatar/parameters/Smile_R", value)
    print(f"Sent right corner: {value:.2f}")

def send_eyes_x(value):
    value = max(0, min(1, value))
    client.send_message("/avatar/parameters/eye_x", value)
    print(f"Sent eyes X position: {value:.2f}")

def send_eyes_y(value):
    value = max(0, min(1, value))
    client.send_message("/avatar/parameters/eye_y", value)
    print(f"Sent eyes Y position: {value:.2f}")

def send_left_eye_blink(value):
    value = max(0, min(1, value))
    client.send_message("/avatar/parameters/Blink_L", value)
    print(f"Sent left eye blink: {value:.2f}")

def send_right_eye_blink(value):
    value = max(0, min(1, value))
    client.send_message("/avatar/parameters/Blink_R", value)
    print(f"Sent right eye blink: {value:.2f}")

def send_brow_frown(value):
    value = max(0, min(1, value))
    client.send_message("/avatar/parameters/Brow_frown", value)
    print(f"Sent brow frown: {value:.2f}")

def send_brow_raise(value):
    value = max(0, min(1, value))
    client.send_message("/avatar/parameters/Brow_raise", value)
    print(f"Sent brow raise: {value:.2f}")

def send_disable_eye():
    value = True
    client.send_message("/avatar/parameters/DisableEyeTracking", value)

# Ініціалізація MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)


# Функція обчислення відстані між двома точками
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Запуск відео
cap = cv2.VideoCapture(0)


try:
    # Для калібрування
    calibration_distances = []
    calibration_widths = []
    # Для калібрування куточків рота
    calibration_left_distances = []
    calibration_right_distances = []
    # Для відношення висоти до ширини в нейтральному положенні
    neutral_ratio = 0.5

    calibration_phase = True
    calibration_counter = 0

    # Для згладжування
    last_mouth_value = 0
    last_left_corner_value = 0
    last_right_corner_value = 0
    last_mouth_o_value = 0
    # Останні дані для ковзного середнього
    mouth_values_history = deque(maxlen=HISTORY_SIZE)
    left_corner_history = deque(maxlen=HISTORY_SIZE)
    right_corner_history = deque(maxlen=HISTORY_SIZE)
    mouth_o_history = deque(maxlen=HISTORY_SIZE)

    # 0.5 це погляд прямо
    last_eyes_x_value = 0.5
    last_eyes_y_value = 0.5

    eyes_x_history = deque(maxlen=EYE_HISTORY_SIZE)
    eyes_y_history = deque(maxlen=EYE_HISTORY_SIZE)

    # Значення калібрування для очей
    calibration_eyes_center_x = 0
    calibration_eyes_center_y = 0
    calibration_eye_width = 0
    calibration_eye_height = 0
    calibration_face_width = 0
    calibration_face_height = 0

    # Кліпання очей
    last_left_blink_value = 0
    last_right_blink_value = 0
    left_blink_history = deque(maxlen=BLINK_HISTORY_SIZE)
    right_blink_history = deque(maxlen=BLINK_HISTORY_SIZE)

    # Брови
    last_brow_frown_value = 0
    last_brow_raise_value = 0
    brow_frown_history = deque(maxlen=BROW_HISTORY_SIZE)
    brow_raise_history = deque(maxlen=BROW_HISTORY_SIZE)

    # Значення калібрування для брів
    calibration_left_brow_inner_y = 0
    calibration_right_brow_inner_y = 0
    calibration_left_brow_inner_x = 0
    calibration_right_brow_inner_x = 0
    calibration_left_brow_center_y = 0
    calibration_right_brow_center_y = 0
    calibration_brow_width = 0  # Відстань між центральними точками брів

    send_disable_eye()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Перетворюємо зображення на формат RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Точки для рота
                upper_lip = face_landmarks.landmark[13]  # Верхня губа
                lower_lip = face_landmarks.landmark[14]  # Нижня губа
                upper_lip2 = face_landmarks.landmark[312]  # Додаткова точка верхньої губы
                lower_lip2 = face_landmarks.landmark[87]  # Додаткова точка нижньої губы
                left_corner = face_landmarks.landmark[61]  # Лівий куточок рота
                right_corner = face_landmarks.landmark[291]  # Правий куточок рота

                # Центральні точки (для виміру відхилення)
                center_top = face_landmarks.landmark[0]  # Верхня центральна точка обличчя
                center_bottom = face_landmarks.landmark[17]  # Нижня центральна точка обличчя

                # Точки для очей
                left_eye_center = face_landmarks.landmark[468]  # Центр лівого ока (зіниця)
                right_eye_center = face_landmarks.landmark[473]  # Центр правого ока (зіниця)
                left_eye_outer = face_landmarks.landmark[263]  # Зовнішній кут лівого ока
                left_eye_inner = face_landmarks.landmark[362]  # Внутрішній кут лівого ока
                right_eye_outer = face_landmarks.landmark[33]  # Зовнішній кут правого ока
                right_eye_inner = face_landmarks.landmark[133]  # Внутрішній кут правого ока

                # Точки для верхніх і нижніх повік
                left_eye_top = face_landmarks.landmark[386]  # Верхня точка лівого ока
                left_eye_bottom = face_landmarks.landmark[374]  # Нижня точка лівого ока
                right_eye_top = face_landmarks.landmark[159]  # Верхня точка правого ока
                right_eye_bottom = face_landmarks.landmark[145]  # Нижня точка правого ока

                # Точки для визначення розмірів обличчя (для нормалізації)
                face_top = face_landmarks.landmark[10]  # Верхня точка обличчя
                face_bottom = face_landmarks.landmark[152]  # Нижня точка обличчя
                face_left = face_landmarks.landmark[234]  # Ліва точка обличчя
                face_right = face_landmarks.landmark[454]  # Права точка обличчя

                # Точки для брів
                left_brow_inner = face_landmarks.landmark[336]  # Внутрішня точка лівої брови
                right_brow_inner = face_landmarks.landmark[107]  # Внутрішня точка правої брови
                left_brow_center = face_landmarks.landmark[296]  # Центральна точка лівої брови
                right_brow_center = face_landmarks.landmark[66]  # Центральна точка правої брови

                # Перетворимо нормалізовані координати на пікселі
                h, w, _ = frame.shape

                # Перетворюємо нормалізовані координати очей в пікселі
                x_left_eye = int(left_eye_center.x * w)
                y_left_eye = int(left_eye_center.y * h)
                x_right_eye = int(right_eye_center.x * w)
                y_right_eye = int(right_eye_center.y * h)

                # Точки зовнішніх кутів очей
                x_left_eye_outer = int(left_eye_outer.x * w)
                y_left_eye_outer = int(left_eye_outer.y * h)
                x_left_eye_inner = int(left_eye_inner.x * w)
                y_left_eye_inner = int(left_eye_inner.y * h)
                x_right_eye_outer = int(right_eye_outer.x * w)
                y_right_eye_outer = int(right_eye_outer.y * h)
                x_right_eye_inner = int(right_eye_inner.x * w)
                y_right_eye_inner = int(right_eye_inner.y * h)

                # Точки верхніх і нижніх повік
                x_left_eye_top = int(left_eye_top.x * w)
                y_left_eye_top = int(left_eye_top.y * h)
                x_left_eye_bottom = int(left_eye_bottom.x * w)
                y_left_eye_bottom = int(left_eye_bottom.y * h)
                x_right_eye_top = int(right_eye_top.x * w)
                y_right_eye_top = int(right_eye_top.y * h)
                x_right_eye_bottom = int(right_eye_bottom.x * w)
                y_right_eye_bottom = int(right_eye_bottom.y * h)

                # Точки обличчя для нормалізації
                x_face_top = int(face_top.x * w)
                y_face_top = int(face_top.y * h)
                x_face_bottom = int(face_bottom.x * w)
                y_face_bottom = int(face_bottom.y * h)
                x_face_left = int(face_left.x * w)
                y_face_left = int(face_left.y * h)
                x_face_right = int(face_right.x * w)
                y_face_right = int(face_right.y * h)

                # Точки брів
                x_left_brow_inner = int(left_brow_inner.x * w)
                y_left_brow_inner = int(left_brow_inner.y * h)
                x_right_brow_inner = int(right_brow_inner.x * w)
                y_right_brow_inner = int(right_brow_inner.y * h)
                x_left_brow_center = int(left_brow_center.x * w)
                y_left_brow_center = int(left_brow_center.y * h)
                x_right_brow_center = int(right_brow_center.x * w)
                y_right_brow_center = int(right_brow_center.y * h)

                # ---- блок очей ----

                # Розрахунок центру обох очей
                eyes_center_x = (x_left_eye + x_right_eye) / 2
                eyes_center_y = (y_left_eye + y_right_eye) / 2

                # Розрахунок ширини та висоти ока (використовуємо середнє значення обох очей)
                left_eye_width = calculate_distance((x_left_eye_inner, y_left_eye_inner),
                                                    (x_left_eye_outer, y_left_eye_outer))
                right_eye_width = calculate_distance((x_right_eye_inner, y_right_eye_inner),
                                                     (x_right_eye_outer, y_right_eye_outer))
                eye_width = (left_eye_width + right_eye_width) / 2

                left_eye_height = calculate_distance((x_left_eye_top, y_left_eye_top),
                                                     (x_left_eye_bottom, y_left_eye_bottom))
                right_eye_height = calculate_distance((x_right_eye_top, y_right_eye_top),
                                                      (x_right_eye_bottom, y_right_eye_bottom))
                eye_height = (left_eye_height + right_eye_height) / 2

                # Розрахунок моргання (співвідношення висоти до ширини)
                left_eye_aspect_ratio = calculate_distance((x_left_eye_top, y_left_eye_top),
                                                           (x_left_eye_bottom, y_left_eye_bottom)) / left_eye_width
                right_eye_aspect_ratio = calculate_distance((x_right_eye_top, y_right_eye_top),
                                                            (x_right_eye_bottom, y_right_eye_bottom)) / right_eye_width

                # Розрахунок ширини та висоти обличчя
                face_width = calculate_distance((x_face_left, y_face_left), (x_face_right, y_face_right))
                face_height = calculate_distance((x_face_top, y_face_top), (x_face_bottom, y_face_bottom))

                # ---- блок рота ----
                # Координати для відкриття рота
                x_upper, y_upper = int(upper_lip.x * w), int(upper_lip.y * h)
                x_lower, y_lower = int(lower_lip.x * w), int(lower_lip.y * h)
                x_upper2, y_upper2 = int(upper_lip2.x * w), int(upper_lip2.y * h)
                x_lower2, y_lower2 = int(lower_lip2.x * w), int(lower_lip2.y * h)

                # Координати для куточків рота
                x_left, y_left = int(left_corner.x * w), int(left_corner.y * h)
                x_right, y_right = int(right_corner.x * w), int(right_corner.y * h)

                # Координати центральної лінії
                x_center_top, y_center_top = int(center_top.x * w), int(center_top.y * h)
                x_center_bottom, y_center_bottom = int(center_bottom.x * w), int(center_bottom.y * h)

                # Центральна вертикальна лінія обличчя
                center_x = (x_center_top + x_center_bottom) / 2

                # Розрахунок ширини та висоти рота
                distance1 = calculate_distance((x_upper, y_upper), (x_lower, y_lower))
                distance2 = calculate_distance((x_upper2, y_upper2), (x_lower2, y_lower2))
                mouth_height = (distance1 + distance2) / 2
                mouth_width = calculate_distance((x_left, y_left), (x_right, y_right))

                # Розрахунок горизонтальної відстані від центру до куточків рота
                left_distance = center_x - x_left
                right_distance = x_right - center_x


                # Фаза калібрування
                if calibration_phase:
                    calibration_distances.append(mouth_height)
                    calibration_widths.append(mouth_width)
                    calibration_left_distances.append(left_distance)
                    calibration_right_distances.append(right_distance)

                    calibration_eyes_center_x = eyes_center_x
                    calibration_eyes_center_y = eyes_center_y
                    calibration_eye_width = eye_width
                    calibration_eye_height = eye_height
                    calibration_face_width = face_width
                    calibration_face_height = face_height

                    calibration_left_brow_inner_y = y_left_brow_inner
                    calibration_right_brow_inner_y = y_right_brow_inner
                    calibration_left_brow_inner_x = x_left_brow_inner
                    calibration_right_brow_inner_x = x_right_brow_inner
                    calibration_left_brow_center_y = y_left_brow_center
                    calibration_right_brow_center_y = y_right_brow_center
                    calibration_brow_width = calculate_distance(
                        (x_left_brow_center, y_left_brow_center),
                        (x_right_brow_center, y_right_brow_center)
                    )


                    calibration_counter += 1
                    cv2.putText(frame, f"Calibration: {calibration_counter}/50", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if calibration_counter >= 50:
                        calibration_phase = False

                        # Калібрування рота
                        calibration_distances.sort()
                        MIN_MOUTH_DISTANCE = sum(calibration_distances[:25]) / 25
                        MAX_MOUTH_DISTANCE = MIN_MOUTH_DISTANCE * 2.5
                        calibration_widths.sort()
                        MIN_MOUTH_WIDTH = sum(calibration_widths[:25]) / 25
                        MAX_MOUTH_WIDTH = MIN_MOUTH_WIDTH * 1.5

                        # Нейтральне співвідношення висоти до ширини
                        neutral_ratio = MIN_MOUTH_DISTANCE / MIN_MOUTH_WIDTH

                        # Калібрування лівого куточка рота
                        calibration_left_distances.sort()
                        MIN_LEFT_CORNER_DISTANCE = sum(calibration_left_distances[:25]) / 25
                        # Зменшуємо максимальне значення для збільшення чутливості
                        MAX_LEFT_CORNER_DISTANCE = MIN_LEFT_CORNER_DISTANCE * 1.2

                        # Калібрування правого куточка рота
                        calibration_right_distances.sort()
                        MIN_RIGHT_CORNER_DISTANCE = sum(calibration_right_distances[:25]) / 25
                        # Зменшуємо максимальне значення для збільшення чутливості
                        MAX_RIGHT_CORNER_DISTANCE = MIN_RIGHT_CORNER_DISTANCE * 1.2

                        print(f"Калібровка завершена.")
                        print(f"Відкриття рота - Мін: {MIN_MOUTH_DISTANCE}, Макс: {MAX_MOUTH_DISTANCE}")
                        print(
                            f"Ширина рота - Мін: {MIN_MOUTH_WIDTH}, Макс: {MAX_MOUTH_WIDTH}, Нейтральное соотношение: {neutral_ratio}")
                        print(f"Лівий куточок - Мін: {MIN_LEFT_CORNER_DISTANCE}, Макс: {MAX_LEFT_CORNER_DISTANCE}")
                        print(f"Правий куточок - Мін: {MIN_RIGHT_CORNER_DISTANCE}, Макс: {MAX_RIGHT_CORNER_DISTANCE}")

                        print(f"Очі - X центр: {calibration_eyes_center_x}, Y центр: {calibration_eyes_center_y}")
                        print(f"Очі - ширина: {calibration_eye_width}, висота: {calibration_eye_height}")
                        print(f"Обличчя - ширина: {calibration_face_width}, висота: {calibration_face_height}")
                else:
                    # Нормалізація відкриття рота
                    normalized_distance = (mouth_height - MIN_MOUTH_DISTANCE) / (
                            MAX_MOUTH_DISTANCE - MIN_MOUTH_DISTANCE)
                    raw_mouth_openness = max(0, min(1, normalized_distance))

                    # Нормалізація ширини рота
                    normalized_width = (mouth_width - MIN_MOUTH_WIDTH) / (MAX_MOUTH_WIDTH - MIN_MOUTH_WIDTH)

                    # Нормалізація лівого і правого куточків рота
                    normalized_left = (left_distance - MIN_LEFT_CORNER_DISTANCE) / (
                            MAX_LEFT_CORNER_DISTANCE - MIN_LEFT_CORNER_DISTANCE)
                    raw_left_corner = max(0, min(1, normalized_left))


                    normalized_right = (right_distance - MIN_RIGHT_CORNER_DISTANCE) / (
                            MAX_RIGHT_CORNER_DISTANCE - MIN_RIGHT_CORNER_DISTANCE)
                    raw_right_corner = max(0, min(1, normalized_right))

                    # співвідношення висоти до ширини
                    current_ratio = mouth_height / mouth_width if mouth_width > 0 else neutral_ratio

                    smile_strength = (raw_left_corner + raw_right_corner) / 2
                    smile_ratio = current_ratio / neutral_ratio
                    if smile_ratio < 0.8 and smile_strength > 0.3:
                        # Зменшення значення відкриття рота пропорційно силі посмішки
                        compensation = smile_strength * SMILE_COMPENSATION * (1 - smile_ratio)
                        raw_mouth_openness = max(0, raw_mouth_openness - compensation)

                    # Згладжування для відкриття рота
                    mouth_openness = SMOOTHING_FACTOR * raw_mouth_openness + (1 - SMOOTHING_FACTOR) * last_mouth_value
                    last_mouth_value = mouth_openness

                    mouth_values_history.append(raw_mouth_openness)
                    moving_avg = sum(mouth_values_history) / len(mouth_values_history)
                    mouth_openness = 0.7 * mouth_openness + 0.3 * moving_avg
                    mouth_openness = math.pow(mouth_openness, 0.8)

                    # Згладжування для лівого куточка
                    left_corner = SMOOTHING_FACTOR * raw_left_corner + (1 - SMOOTHING_FACTOR) * last_left_corner_value
                    last_left_corner_value = left_corner

                    left_corner_history.append(raw_left_corner)
                    left_moving_avg = sum(left_corner_history) / len(left_corner_history)
                    left_corner = 0.7 * left_corner + 0.3 * left_moving_avg

                    # збільшення маленьких значень
                    left_corner = math.pow(left_corner, SMILE_POWER)
                    left_corner = min(1.0, left_corner * SMILE_SCALE)

                    # Згладжування для правого куточка
                    right_corner = SMOOTHING_FACTOR * raw_right_corner + (
                            1 - SMOOTHING_FACTOR) * last_right_corner_value
                    last_right_corner_value = right_corner

                    right_corner_history.append(raw_right_corner)
                    right_moving_avg = sum(right_corner_history) / len(right_corner_history)
                    right_corner = 0.7 * right_corner + 0.3 * right_moving_avg

                    # збільшення маленьких значень
                    right_corner = math.pow(right_corner, SMILE_POWER)
                    right_corner = min(1.0, right_corner * SMILE_SCALE)

                    if left_corner > 0.9 and right_corner > 0.9 and mouth_openness < 0.3:
                        mouth_openness = 0

                    # -- форма "О" --
                    o_shape_score = 0.0
                    # Оцениваем форму губ для звука "О"
                    if mouth_openness > 0.1:  # Рот должен быть открыт
                        # форма рота має бути круглою
                        ratio_factor = min(1.0, (current_ratio / (neutral_ratio * O_SHAPE_RATIO_THRESHOLD)))
                        # куточки рота не мають бути сильно підняті
                        corners_factor = 1.0 - min(1.0, (left_corner + right_corner) / (2 * O_SHAPE_CORNERS_THRESHOLD))
                        # Фінальна оцінка
                        o_shape_score = ratio_factor * corners_factor * mouth_openness

                    # Згладжування для форми "О"
                    raw_mouth_o = max(0, min(1, o_shape_score))
                    mouth_o = SMOOTHING_FACTOR * raw_mouth_o + (1 - SMOOTHING_FACTOR) * last_mouth_o_value
                    last_mouth_o_value = mouth_o

                    mouth_o_history.append(raw_mouth_o)
                    o_moving_avg = sum(mouth_o_history) / len(mouth_o_history)
                    mouth_o = 0.7 * mouth_o + 0.3 * o_moving_avg

                    # Заборона одночасного значення 1 для відкриття рота та форми "О"
                    if mouth_openness + mouth_o > MAX_COMBINED_VALUE:
                        mouth_openness = max(0, MAX_COMBINED_VALUE - mouth_o)

                    send_mouth_openness(mouth_openness)
                    send_mouth_o_shape(mouth_o)
                    send_left_corner(left_corner)
                    send_right_corner(right_corner)

                    # ---- блок очей ----

                    # Нормалізуємо зсув очей відносно ширини обличчя та ока
                    eye_movement_range_x = calibration_face_width / 8  # Діапазон руху очей по горизонталі
                    eye_movement_range_y = calibration_face_height / 10  # Діапазон руху очей по вертикалі

                    # Різниця відносно калібрування
                    x_diff = eyes_center_x - calibration_eyes_center_x
                    y_diff = eyes_center_y - calibration_eyes_center_y

                    # Нормалізація до діапазону [-1, 1]
                    normalized_x = np.clip(x_diff / eye_movement_range_x, -1, 1)
                    normalized_y = np.clip(y_diff / eye_movement_range_y, -1, 1)

                    # Перетворення до діапазону [0, 1], де 0.5 - центр
                    raw_eyes_x = 0.5 - normalized_x * 0.5 * EYE_SCALE
                    raw_eyes_y = 0.5 - normalized_y * 0.5 * EYE_SCALE

                    # Згладжування по горизонталі
                    eyes_x = EYE_SMOOTHING_FACTOR * raw_eyes_x + (1 - EYE_SMOOTHING_FACTOR) * last_eyes_x_value
                    last_eyes_x_value = eyes_x

                    eyes_x_history.append(raw_eyes_x)
                    x_moving_avg = sum(eyes_x_history) / len(eyes_x_history)
                    eyes_x = 0.7 * eyes_x + 0.3 * x_moving_avg

                    # Згладжування по вертикалі
                    eyes_y = EYE_SMOOTHING_FACTOR * raw_eyes_y + (1 - EYE_SMOOTHING_FACTOR) * last_eyes_y_value
                    last_eyes_y_value = eyes_y

                    eyes_y_history.append(raw_eyes_y)
                    y_moving_avg = sum(eyes_y_history) / len(eyes_y_history)
                    eyes_y = 0.7 * eyes_y + 0.3 * y_moving_avg

                    eyes_x = max(0, min(1, eyes_x))
                    eyes_y = max(0, min(1, eyes_y))

                    # Інвертуємо X, бо камера дзеркально відображає рух
                    eyes_x = abs(eyes_x - 1)

                    send_eyes_x(eyes_x)
                    send_eyes_y(eyes_y)

                    # -- кліпання --
                    raw_left_blink = 1.0 - min(1.0, left_eye_aspect_ratio / BLINK_THRESHOLD)
                    raw_right_blink = 1.0 - min(1.0, right_eye_aspect_ratio / BLINK_THRESHOLD)

                    # Згладжування кліпання лівого ока
                    left_blink = BLINK_SMOOTHING_FACTOR * raw_left_blink + (
                                1 - BLINK_SMOOTHING_FACTOR) * last_left_blink_value
                    last_left_blink_value = left_blink

                    left_blink_history.append(raw_left_blink)
                    left_blink_avg = sum(left_blink_history) / len(left_blink_history)
                    left_blink = 0.7 * left_blink + 0.3 * left_blink_avg

                    # Згладжування кліпання правого ока
                    right_blink = BLINK_SMOOTHING_FACTOR * raw_right_blink + (
                                1 - BLINK_SMOOTHING_FACTOR) * last_right_blink_value
                    last_right_blink_value = right_blink

                    right_blink_history.append(raw_right_blink)
                    right_blink_avg = sum(right_blink_history) / len(right_blink_history)
                    right_blink = 0.7 * right_blink + 0.3 * right_blink_avg

                    send_left_eye_blink(left_blink)
                    send_right_eye_blink(right_blink)

                    # ---- блок брів ----

                    # Відстань між внутрішніми точками брів
                    current_inner_distance = calculate_distance(
                        (x_left_brow_inner, y_left_brow_inner),
                        (x_right_brow_inner, y_right_brow_inner)
                    )
                    # -- хмурість --
                    # Опускання внутрішніх кутів
                    left_inner_y_diff = y_left_brow_inner - calibration_left_brow_inner_y
                    right_inner_y_diff = y_right_brow_inner - calibration_right_brow_inner_y

                    # Звуження внутрішніх кутів
                    left_inner_x_diff = calibration_left_brow_inner_x - x_left_brow_inner
                    right_inner_x_diff = x_right_brow_inner - calibration_right_brow_inner_x

                    # Ознаки хмурості двох брів
                    horizontal_squeeze = (left_inner_x_diff + right_inner_x_diff) / (calibration_brow_width * 0.2)
                    vertical_drop = (left_inner_y_diff + right_inner_y_diff) / (calibration_face_height * 0.05)

                    # Фінальна оцінка
                    raw_brow_frown = max(0, min(1, (horizontal_squeeze * 0.7 + vertical_drop * 0.3) * BROW_FROWN_SCALE))

                    # -- здивування --
                    # Підняття брів
                    left_center_y_diff = calibration_left_brow_center_y - y_left_brow_center
                    right_center_y_diff = calibration_right_brow_center_y - y_right_brow_center

                    # Середнє значення 2 брів
                    brow_raise_raw = (left_center_y_diff + right_center_y_diff) / (calibration_face_height * 0.08)
                    raw_brow_raise = max(0, min(1, brow_raise_raw * BROW_RAISE_SCALE))

                    # Якщо виявлено хмуріння, зменшується підняття брів
                    if raw_brow_frown > 0.3:
                        raw_brow_raise = max(0, raw_brow_raise - raw_brow_frown * 0.5)

                    # Згладжування хмурості
                    brow_frown = BROW_SMOOTHING_FACTOR * raw_brow_frown + (
                                1 - BROW_SMOOTHING_FACTOR) * last_brow_frown_value
                    last_brow_frown_value = brow_frown

                    brow_frown_history.append(raw_brow_frown)
                    frown_avg = sum(brow_frown_history) / len(brow_frown_history)
                    brow_frown = 0.7 * brow_frown + 0.3 * frown_avg

                    # Згладжування здивування
                    brow_raise = BROW_SMOOTHING_FACTOR * raw_brow_raise + (
                                1 - BROW_SMOOTHING_FACTOR) * last_brow_raise_value
                    last_brow_raise_value = brow_raise

                    brow_raise_history.append(raw_brow_raise)
                    raise_avg = sum(brow_raise_history) / len(brow_raise_history)
                    brow_raise = 0.7 * brow_raise + 0.3 * raise_avg

                    send_brow_frown(brow_frown)
                    send_brow_raise(brow_raise)


                    # ---- Відображення значень на екрані ----
                    cv2.putText(frame, f"Eyes X: {eyes_x:.2f}", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                    cv2.putText(frame, f"Eyes Y: {eyes_y:.2f}", (50, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                    cv2.putText(frame, f"Mouth: {mouth_openness:.2f}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"O-Shape: {mouth_o:.2f}", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                    cv2.putText(frame, f"Left: {left_corner:.2f}", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.putText(frame, f"Right: {right_corner:.2f}", (50, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    #cv2.putText(frame, f"Ratio: {current_ratio / neutral_ratio:.2f}", (50, 170),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame, f"Left Blink: {left_blink:.2f}", (50, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 200), 2)
                    cv2.putText(frame, f"Right Blink: {right_blink:.2f}", (50, 290),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 200), 2)

                    cv2.putText(frame, f"Brow Frown: {brow_frown:.2f}", (50, 320),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 200, 255), 2)
                    cv2.putText(frame, f"Brow Raise: {brow_raise:.2f}", (50, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 200, 255), 2)

                # ---- Візуалізація точек ----
                # Точки губ для відкриття рота
                cv2.circle(frame, (x_upper, y_upper), 3, (0, 255, 0), -1)
                cv2.circle(frame, (x_lower, y_lower), 3, (0, 255, 0), -1)
                cv2.circle(frame, (x_upper2, y_upper2), 3, (0, 255, 0), -1)
                cv2.circle(frame, (x_lower2, y_lower2), 3, (0, 255, 0), -1)

                # Крапки куточків рота
                cv2.circle(frame, (x_left, y_left), 5, (255, 0, 0), -1)
                cv2.circle(frame, (x_right, y_right), 5, (0, 0, 255), -1)

                # Центральна лінія обличчя
                cv2.line(frame, (int(center_x), y_center_top), (int(center_x), y_center_bottom), (255, 255, 0), 1)

                # Лінія для позначення ширини рота
                cv2.line(frame, (x_left, y_left), (x_right, y_right), (0, 255, 255), 1)

                # Центри очей
                cv2.circle(frame, (x_left_eye, y_left_eye), 3, (0, 255, 255), -1)
                cv2.circle(frame, (x_right_eye, y_right_eye), 3, (0, 255, 255), -1)

                # Кути очей та контур
                cv2.circle(frame, (x_left_eye_inner, y_left_eye_inner), 2, (0, 165, 255), -1)
                cv2.circle(frame, (x_left_eye_outer, y_left_eye_outer), 2, (0, 165, 255), -1)
                cv2.circle(frame, (x_left_eye_top, y_left_eye_top), 2, (0, 165, 255), -1)
                cv2.circle(frame, (x_left_eye_bottom, y_left_eye_bottom), 2, (0, 165, 255), -1)

                cv2.circle(frame, (x_right_eye_inner, y_right_eye_inner), 2, (0, 165, 255), -1)
                cv2.circle(frame, (x_right_eye_outer, y_right_eye_outer), 2, (0, 165, 255), -1)
                cv2.circle(frame, (x_right_eye_top, y_right_eye_top), 2, (0, 165, 255), -1)
                cv2.circle(frame, (x_right_eye_bottom, y_right_eye_bottom), 2, (0, 165, 255), -1)

                # Контур лівого ока
                cv2.line(frame, (x_left_eye_inner, y_left_eye_inner), (x_left_eye_top, y_left_eye_top), (0, 165, 255),
                         1)
                cv2.line(frame, (x_left_eye_top, y_left_eye_top), (x_left_eye_outer, y_left_eye_outer), (0, 165, 255),
                         1)
                cv2.line(frame, (x_left_eye_outer, y_left_eye_outer), (x_left_eye_bottom, y_left_eye_bottom),
                         (0, 165, 255), 1)
                cv2.line(frame, (x_left_eye_bottom, y_left_eye_bottom), (x_left_eye_inner, y_left_eye_inner),
                         (0, 165, 255), 1)

                # Контур правого ока
                cv2.line(frame, (x_right_eye_inner, y_right_eye_inner), (x_right_eye_top, y_right_eye_top),
                         (0, 165, 255), 1)
                cv2.line(frame, (x_right_eye_top, y_right_eye_top), (x_right_eye_outer, y_right_eye_outer),
                         (0, 165, 255), 1)
                cv2.line(frame, (x_right_eye_outer, y_right_eye_outer), (x_right_eye_bottom, y_right_eye_bottom),
                         (0, 165, 255), 1)
                cv2.line(frame, (x_right_eye_bottom, y_right_eye_bottom), (x_right_eye_inner, y_right_eye_inner),
                         (0, 165, 255), 1)

                # Точки брів
                cv2.circle(frame, (x_left_brow_inner, y_left_brow_inner), 3, (150, 200, 255), -1)
                cv2.circle(frame, (x_right_brow_inner, y_right_brow_inner), 3, (150, 200, 255), -1)
                cv2.circle(frame, (x_left_brow_center, y_left_brow_center), 3, (150, 200, 255), -1)
                cv2.circle(frame, (x_right_brow_center, y_right_brow_center), 3, (150, 200, 255), -1)
                cv2.line(frame, (x_left_brow_inner, y_left_brow_inner),
                         (x_left_brow_center, y_left_brow_center), (150, 200, 255), 1)
                cv2.line(frame, (x_right_brow_inner, y_right_brow_inner),
                         (x_right_brow_center, y_right_brow_center), (150, 200, 255), 1)

        # Відображення камери
        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Script stopped.")

finally:
    cap.release()
    cv2.destroyAllWindows()