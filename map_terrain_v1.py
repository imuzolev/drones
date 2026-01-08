"""
Полет квадрокоптера по периметру карты в AirSim.
Автор: AI Assistant
Дата: 2026

Особенности:
- Атомарная запись settings.json (защита от повреждения)
- Безопасный полёт с обходом препятствий
- Использование всех доступных радаров и сенсоров для навигации
- Полет по периметру по часовой стрелке на высоте 1 метр
- Отображение координат в симуляторе
- Визуализация маршрута в конце полета
"""

import airsim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import time
import sys
import math
import subprocess
import shutil
import tempfile
from datetime import datetime

# =========================
# КОНФИГУРАЦИЯ
# =========================

# Имя дрона в AirSim
VEHICLE_NAME = "SimpleFlight"

# Путь к settings.json
AIRSIM_SETTINGS_PATH = os.path.expanduser(r"~\Documents\AirSim\settings.json")

# Параметры полета по периметру
PERIMETER_SIZE_X = 50      # размер области по X (метры)
PERIMETER_SIZE_Y = 50      # размер области по Y (метры)
FLIGHT_HEIGHT = -1.0       # высота полета 1 метр (NED, отрицательное = вверх)
PERIMETER_OFFSET = 2.0     # отступ от края карты (метры)

# Параметры безопасности полёта
CMD_HZ = 20.0              # частота команд управления
CMD_DT = 1.0 / CMD_HZ
SPEED_KMH = 10.0           # скорость 50 км/ч (уменьшено в 2 раза)
SPEED_MS = SPEED_KMH / 3.6 # скорость в м/с (~13.89 м/с)
MAX_SPEED = SPEED_MS       # максимальная скорость
MIN_SPEED = 0.3            # минимальная скорость при обходе препятствий (уменьшено для лучшего торможения)
ARRIVAL_TOL = 1.5          # допуск прибытия в точку

# Пороги безопасности
SAFE_DISTANCE = 4.0        # начинаем замедление
WARN_DISTANCE = 2.0        # начинаем обход
STOP_DISTANCE = 1.0        # аварийный стоп/подъём

# Параметры обхода препятствий
AVOID_LATERAL_SPEED = 1.5
AVOID_VERTICAL_SPEED = 1.2
MAX_AVOID_ITERATIONS = 50

# Depth камера
DEPTH_CENTER_FRACTION = 0.3
DEPTH_MAX_RANGE = 50.0

# =========================
# НАСТРОЙКА КОНСОЛИ WINDOWS
# =========================

if sys.platform == 'win32':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        pass

# =========================
# УТИЛИТЫ
# =========================

def log_info(msg):
    """Вывод информационного сообщения."""
    print(f"[INFO] {msg}")

def log_warn(msg):
    """Вывод предупреждения."""
    print(f"[WARN] {msg}")

def log_error(msg):
    """Вывод ошибки."""
    print(f"[ERROR] {msg}")

def log_ok(msg):
    """Вывод успешного сообщения."""
    print(f"[OK] {msg}")

# =========================
# НАСТРОЙКИ AIRSIM (АТОМАРНАЯ ЗАПИСЬ)
# =========================

def get_correct_settings():
    """Возвращает правильные настройки AirSim с лидаром."""
    return {
        "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "Vehicles": {
            VEHICLE_NAME: {
                "VehicleType": "SimpleFlight",
                "AutoCreate": True,
                "Sensors": {
                    "DistanceFront": {
                        "SensorType": 5,
                        "Enabled": True,
                        "MinDistance": 0.2,
                        "MaxDistance": 40,
                        "X": 0.2, "Y": 0.0, "Z": 0.0,
                        "Yaw": 0, "Pitch": 0, "Roll": 0
                    },
                    "DistanceBack": {
                        "SensorType": 5,
                        "Enabled": True,
                        "MinDistance": 0.2,
                        "MaxDistance": 40,
                        "X": -0.2, "Y": 0.0, "Z": 0.0,
                        "Yaw": 180, "Pitch": 0, "Roll": 0
                    },
                    "DistanceLeft": {
                        "SensorType": 5,
                        "Enabled": True,
                        "MinDistance": 0.2,
                        "MaxDistance": 40,
                        "X": 0.0, "Y": 0.2, "Z": 0.0,
                        "Yaw": 90, "Pitch": 0, "Roll": 0
                    },
                    "DistanceRight": {
                        "SensorType": 5,
                        "Enabled": True,
                        "MinDistance": 0.2,
                        "MaxDistance": 40,
                        "X": 0.0, "Y": -0.2, "Z": 0.0,
                        "Yaw": -90, "Pitch": 0, "Roll": 0
                    },
                    "LidarFront": {
                        "SensorType": 6,
                        "Enabled": True,
                        "NumberOfChannels": 16,
                        "RotationsPerSecond": 10,
                        "PointsPerSecond": 20000,
                        "Range": 30,
                        "HorizontalFOVStart": -60,
                        "HorizontalFOVEnd": 60,
                        "VerticalFOVUpper": 10,
                        "VerticalFOVLower": -25,
                        "X": 0.2, "Y": 0.0, "Z": 0.0,
                        "DrawDebugPoints": True
                    }
                }
            }
        }
    }


def write_settings_atomic(settings_dict, filepath):
    """
    Атомарная запись settings.json.
    Записывает во временный файл, затем переименовывает.
    Это гарантирует, что файл никогда не будет повреждён.
    """
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)
    
    # Создаём временный файл в той же директории (важно для rename)
    fd, temp_path = tempfile.mkstemp(suffix='.json', dir=directory)
    
    try:
        # Записываем данные во временный файл
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=2, ensure_ascii=False)
        
        # Атомарно переименовываем (на Windows это replace)
        # Сначала удаляем старый файл, если он существует
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Переименовываем временный файл
        shutil.move(temp_path, filepath)
        
        return True
        
    except Exception as e:
        # Если что-то пошло не так, удаляем временный файл
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise e


def validate_settings(filepath):
    """
    Проверяет, что settings.json содержит правильные настройки.
    Возвращает (is_valid, error_message).
    """
    if not os.path.exists(filepath):
        return False, "Файл не существует"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Проверяем, что файл не пустой и не слишком короткий
        if len(content) < 50:
            return False, f"Файл слишком короткий ({len(content)} символов)"
        
        settings = json.loads(content)
        
        # Проверяем основные параметры
        if settings.get("SimMode") != "Multirotor":
            return False, f"Неправильный SimMode: {settings.get('SimMode')}"
        
        # Проверяем наличие Vehicles
        if "Vehicles" not in settings:
            return False, "Отсутствует секция Vehicles"
        
        # Проверяем наличие нашего дрона с лидаром
        vehicle = settings.get("Vehicles", {}).get(VEHICLE_NAME, {})
        sensors = vehicle.get("Sensors", {})
        
        if "LidarFront" not in sensors:
            return False, "Отсутствует LidarFront сенсор"
        
        return True, None
        
    except json.JSONDecodeError as e:
        return False, f"Ошибка JSON: {e}"
    except Exception as e:
        return False, f"Ошибка чтения: {e}"


def ensure_settings():
    """
    Проверяет и при необходимости восстанавливает settings.json.
    Использует атомарную запись для предотвращения повреждения файла.
    """
    log_info(f"Проверка {AIRSIM_SETTINGS_PATH}...")
    
    is_valid, error = validate_settings(AIRSIM_SETTINGS_PATH)
    
    if is_valid:
        log_ok("settings.json корректен (Multirotor, LidarFront)")
        return True
    
    log_warn(f"settings.json некорректен: {error}")
    log_info("Восстанавливаю настройки...")
    
    # Создаём резервную копию, если файл существует
    if os.path.exists(AIRSIM_SETTINGS_PATH):
        backup_path = AIRSIM_SETTINGS_PATH + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copy2(AIRSIM_SETTINGS_PATH, backup_path)
            log_info(f"Резервная копия: {backup_path}")
        except Exception as e:
            log_warn(f"Не удалось создать резервную копию: {e}")
    
    # Записываем правильные настройки атомарно
    try:
        write_settings_atomic(get_correct_settings(), AIRSIM_SETTINGS_PATH)
        log_ok("settings.json восстановлен")
        return True
    except Exception as e:
        log_error(f"Не удалось записать settings.json: {e}")
        return False


# =========================
# ПРОВЕРКА И ЗАПУСК СИМУЛЯТОРА
# =========================

# Пути к симулятору (можно изменить под вашу установку)
UE4_EDITOR_PATH = r"C:\Program Files\Epic Games\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe"
BLOCKS_PROJECT_PATH = r"C:\CORTEXIS\airsim\Unreal\Environments\Blocks\Blocks.uproject"


def is_simulator_running():
    """Проверяет, запущен ли UE4Editor."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq UE4Editor.exe'],
            capture_output=True, text=True, timeout=5
        )
        return 'UE4Editor.exe' in result.stdout
    except Exception:
        return False


def start_simulator():
    """
    Запускает симулятор Blocks, если он ещё не запущен.
    Возвращает True, если симулятор был запущен или уже работал.
    """
    # Проверяем, не запущен ли уже
    if is_simulator_running():
        log_ok("Симулятор уже запущен")
        return True
    
    # Проверяем существование файлов
    if not os.path.exists(UE4_EDITOR_PATH):
        log_warn(f"UE4Editor не найден по пути: {UE4_EDITOR_PATH}")
        log_info("Попробуйте запустить симулятор вручную")
        return False
    
    if not os.path.exists(BLOCKS_PROJECT_PATH):
        log_warn(f"Проект Blocks не найден по пути: {BLOCKS_PROJECT_PATH}")
        log_info("Попробуйте запустить симулятор вручную")
        return False
    
    # Запускаем симулятор
    log_info("Запуск симулятора Blocks...")
    log_info(f"  UE4Editor: {UE4_EDITOR_PATH}")
    log_info(f"  Проект: {BLOCKS_PROJECT_PATH}")
    
    try:
        # Запускаем процесс в фоне
        subprocess.Popen(
            [UE4_EDITOR_PATH, BLOCKS_PROJECT_PATH],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        log_ok("Симулятор запущен, ожидаю загрузки...")
        log_info("Это может занять 30-60 секунд...")
        return True
    except Exception as e:
        log_error(f"Ошибка запуска симулятора: {e}")
        log_info("Попробуйте запустить вручную:")
        log_info(f'  "{UE4_EDITOR_PATH}" "{BLOCKS_PROJECT_PATH}"')
        return False


def wait_for_simulator(timeout=120):
    """
    Ожидает запуска симулятора.
    Если симулятор не запущен, пытается запустить его автоматически.
    """
    if is_simulator_running():
        log_ok("Симулятор уже запущен")
        time.sleep(3)  # Даём время на полную инициализацию
        return True
    
    # Пытаемся запустить автоматически
    log_info("Симулятор не обнаружен, пытаюсь запустить...")
    if start_simulator():
        log_info(f"Ожидание загрузки симулятора (до {timeout} секунд)...")
    else:
        log_warn("Не удалось запустить симулятор автоматически")
        log_info("Пожалуйста, запустите Unreal Engine с проектом Blocks вручную")
        log_info(f"Ожидание {timeout} секунд...")
    
    # Ждём, пока симулятор запустится
    start_time = time.time()
    check_interval = 2
    
    while time.time() - start_time < timeout:
        if is_simulator_running():
            elapsed = int(time.time() - start_time)
            log_ok(f"Симулятор обнаружен! (через {elapsed} секунд)")
            log_info("Ожидаю полной инициализации...")
            time.sleep(5)  # Даём время на полную инициализацию
            return True
        time.sleep(check_interval)
    
    log_warn(f"Симулятор не обнаружен за {timeout} секунд")
    log_info("Продолжаю попытку подключения (возможно, симулятор ещё загружается)...")
    return False  # Не критично, попробуем подключиться всё равно


# =========================
# ПОДКЛЮЧЕНИЕ К AIRSIM
# =========================

def connect_to_airsim(max_retries=30, retry_delay=2.0):
    """
    Подключается к AirSim с повторными попытками.
    Ждёт готовности симулятора перед подключением.
    """
    log_info("Ожидание готовности симулятора AirSim...")
    log_info("Симулятор должен полностью загрузиться (окно должно быть открыто)")
    
    for attempt in range(1, max_retries + 1):
        try:
            client = airsim.MultirotorClient()
            if client.ping():
                log_ok(f"Подключено к AirSim (попытка {attempt}/{max_retries})")
                return client
        except Exception as e:
            if attempt < max_retries:
                print(f"  Попытка {attempt}/{max_retries}: симулятор не готов, жду {retry_delay} сек...")
                print(f"     Ошибка: {type(e).__name__}")
                time.sleep(retry_delay)
            else:
                log_error(f"НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ К AIRSIM!")
                log_error(f"   Ошибка: {e}")
                log_info("\n   РЕШЕНИЕ:")
                log_info("   1. Убедитесь, что симулятор Blocks запущен и полностью загружен")
                log_info("   2. Окно симулятора должно быть открыто")
                log_info("   3. Убедитесь, что AirSim плагин активирован")
                log_info("   4. Попробуйте запустить скрипт снова")
                return None
    
    return None


# =========================
# КЛАСС БЕЗОПАСНОГО ДРОНА
# =========================

class SafeDrone:
    """Класс для безопасного управления дроном с обходом препятствий."""
    
    def __init__(self, client: airsim.MultirotorClient):
        self.client = client
        self.lidar_available = False
        self.last_log_time = 0
        self.log_interval = 2.0
        self.start_position = None  # Стартовая позиция (x, y, z)
        self.route_points = []  # Точки маршрута для визуализации
        self.last_coord_update = 0
        self.coord_update_interval = 0.1  # Обновление координат каждые 100мс
        self.last_speed_display = 0
        self.speed_display_interval = 0.2  # Обновление скорости каждые 200мс
        self.lidar_points = []  # Точки лидара для визуализации
        self.last_lidar_collection = 0
        self.lidar_collection_interval = 0.2  # Сбор точек лидара каждые 200мс
        
    def initialize(self):
        """Инициализация дрона."""
        log_info("Инициализация дрона...")
        
        # Проверяем лидар
        try:
            data = self.client.getLidarData(lidar_name="LidarFront", vehicle_name=VEHICLE_NAME)
            if data and data.point_cloud:
                self.lidar_available = True
                log_ok(f"Лидар доступен ({len(data.point_cloud)//3} точек)")
            else:
                log_warn("Лидар подключен, но нет данных")
        except Exception as e:
            log_warn(f"Лидар недоступен: {e}")
        
        # Проверяем distance сенсоры
        available_sensors = []
        for sensor_name in ["DistanceFront", "DistanceBack", "DistanceLeft", "DistanceRight"]:
            try:
                dist = self.get_distance_sensor(sensor_name)
                if dist is not None:
                    available_sensors.append(sensor_name)
            except Exception:
                pass
        
        if available_sensors:
            log_ok(f"Distance сенсоры доступны: {', '.join(available_sensors)}")
        else:
            log_warn("Distance сенсоры недоступны")
        
        # Включаем API и армим
        self.client.enableApiControl(True, VEHICLE_NAME)
        self.client.armDisarm(True, VEHICLE_NAME)
        
        log_ok("Дрон инициализирован")
        
    def takeoff(self):
        """Безопасный взлёт."""
        log_info("Взлёт...")
        self.client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
        time.sleep(1.0)
        # Сохраняем стартовую позицию
        self.start_position = self.get_position()
        self.route_points = [self.start_position]
        log_ok(f"Взлёт завершён. Стартовая позиция: {self.start_position}")
        
    def land(self):
        """Безопасная посадка."""
        log_info("Посадка...")
        self.client.landAsync(vehicle_name=VEHICLE_NAME).join()
        self.client.armDisarm(False, VEHICLE_NAME)
        self.client.enableApiControl(False, VEHICLE_NAME)
        log_ok("Посадка завершена")
        
    def get_position(self):
        """Получает текущую позицию дрона."""
        state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        p = state.kinematics_estimated.position
        return float(p.x_val), float(p.y_val), float(p.z_val)
    
    def get_velocity(self):
        """Получает текущую скорость дрона (м/с)."""
        try:
            state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
            v = state.kinematics_estimated.linear_velocity
            vx = float(v.x_val)
            vy = float(v.y_val)
            vz = float(v.z_val)
            # Вычисляем общую скорость
            speed = math.sqrt(vx**2 + vy**2 + vz**2)
            return speed, vx, vy, vz
        except Exception:
            return 0.0, 0.0, 0.0, 0.0
    
    def get_yaw(self):
        """Получает текущий yaw (ориентацию) дрона в радианах."""
        try:
            state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
            orientation = state.kinematics_estimated.orientation
            # Преобразуем кватернион в yaw (в радианах)
            import math
            # Yaw из кватерниона: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
            w = orientation.w_val
            x = orientation.x_val
            y = orientation.y_val
            z = orientation.z_val
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            return yaw
        except Exception:
            return 0.0
    
    def rotate_yaw(self, target_yaw_rad, tolerance_rad=0.1):
        """
        Поворачивает дрон до заданного yaw.
        target_yaw_rad: целевой yaw в радианах
        tolerance_rad: допуск в радианах
        """
        log_info(f"Поворот до yaw: {math.degrees(target_yaw_rad):.1f}°")
        
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            current_yaw = self.get_yaw()
            
            # Вычисляем разницу углов (учитываем переход через 0)
            diff = target_yaw_rad - current_yaw
            
            # Нормализуем разницу в диапазон [-pi, pi]
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff < -math.pi:
                diff += 2 * math.pi
            
            # Если достигли цели
            if abs(diff) < tolerance_rad:
                log_ok(f"Поворот завершен. Текущий yaw: {math.degrees(current_yaw):.1f}°")
                self.hover()
                time.sleep(0.5)
                return True
            
            # Вычисляем скорость поворота (пропорционально разнице)
            yaw_rate = np.clip(diff * 2.0, -1.0, 1.0)  # Максимальная скорость поворота
            
            # Поворачиваемся
            self.client.rotateByYawRateAsync(
                yaw_rate,
                duration=CMD_DT,
                vehicle_name=VEHICLE_NAME
            )
            
            time.sleep(CMD_DT)
            iteration += 1
        
        log_warn(f"Поворот не завершен за {max_iterations} итераций")
        return False
    
    def rotate_right_90(self):
        """Поворачивает дрон на 90 градусов вправо (по часовой стрелке)."""
        current_yaw = self.get_yaw()
        target_yaw = current_yaw - math.pi / 2.0  # Минус = поворот вправо (по часовой)
        return self.rotate_yaw(target_yaw)
    
    def has_collision(self):
        """Проверяет столкновение."""
        try:
            info = self.client.simGetCollisionInfo(vehicle_name=VEHICLE_NAME)
            return bool(info.has_collided)
        except Exception:
            return False
    
    def get_depth_distance(self):
        """Получает минимальную дистанцию по depth-камере."""
        try:
            req = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, 
                                       pixels_as_float=True, compress=False)
            resp = self.client.simGetImages([req], vehicle_name=VEHICLE_NAME)[0]
            
            if resp is None or resp.width == 0 or resp.height == 0:
                return None
            
            w, h = int(resp.width), int(resp.height)
            arr = np.array(resp.image_data_float, dtype=np.float32)
            
            if arr.size != w * h:
                return None
            
            depth = arr.reshape(h, w)
            
            # Центральная область
            cw = int(w * DEPTH_CENTER_FRACTION)
            ch = int(h * DEPTH_CENTER_FRACTION)
            x0, x1 = w // 2 - cw // 2, w // 2 + cw // 2
            y0, y1 = h // 2 - ch // 2, h // 2 + ch // 2
            
            roi = depth[y0:y1, x0:x1]
            roi = roi[np.isfinite(roi) & (roi > 0.01) & (roi < DEPTH_MAX_RANGE)]
            
            if roi.size == 0:
                return None
            
            return float(np.min(roi))
            
        except Exception:
            return None
    
    def collect_lidar_points(self):
        """Собирает точки лидара для визуализации."""
        now = time.time()
        if now - self.last_lidar_collection < self.lidar_collection_interval:
            return
        
        self.last_lidar_collection = now
        
        if not self.lidar_available:
            return
        
        try:
            data = self.client.getLidarData(lidar_name="LidarFront", vehicle_name=VEHICLE_NAME)
            
            if not data.point_cloud:
                return
            
            # Получаем текущую позицию и ориентацию дрона
            state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
            position = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            
            # Позиция дрона
            drone_x = float(position.x_val)
            drone_y = float(position.y_val)
            drone_z = float(position.z_val)
            
            # Преобразуем точки лидара в мировые координаты
            pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
            
            # Точки лидара в локальной системе координат лидара
            # Лидар установлен впереди дрона (X=0.2, Y=0, Z=0) относительно центра дрона
            # Для упрощения, учитываем только смещение лидара (0.2, 0, 0)
            # и добавляем позицию дрона
            # В реальности нужна полная трансформация с учетом ориентации, но для визуализации это достаточно
            
            # Смещение лидара от центра дрона (в локальной системе дрона)
            lidar_offset = np.array([0.2, 0.0, 0.0])
            
            # Преобразуем точки: точки лидара + смещение лидара + позиция дрона
            # Упрощенная версия без учета ориентации (для визуализации достаточно)
            world_pts = pts + lidar_offset + np.array([drone_x, drone_y, drone_z])
            
            # Сохраняем подвыборку точек (каждую 10-ю для экономии памяти)
            if len(world_pts) > 0:
                sampled_pts = world_pts[::10]  # Каждая 10-я точка
                self.lidar_points.extend(sampled_pts.tolist())
                
                # Ограничиваем количество точек (последние 50000 точек)
                if len(self.lidar_points) > 50000:
                    self.lidar_points = self.lidar_points[-50000:]
                    
        except Exception as e:
            # Тихая ошибка, чтобы не засорять вывод
            pass
    
    def get_lidar_distance(self):
        """Получает минимальную дистанцию по лидару."""
        if not self.lidar_available:
            return None
            
        try:
            data = self.client.getLidarData(lidar_name="LidarFront", vehicle_name=VEHICLE_NAME)
            
            if not data.point_cloud:
                return None
            
            pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
            
            # Фильтруем точки впереди дрона
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
            mask = (xs > 0.5) & (np.abs(ys) < 3.0) & (np.abs(zs) < 2.0)
            
            if not np.any(mask):
                return None
            
            dists = np.sqrt(xs[mask]**2 + ys[mask]**2 + zs[mask]**2)
            return float(np.min(dists))
            
        except Exception:
            return None
    
    def get_distance_sensor(self, sensor_name):
        """Получает дистанцию от distance сенсора."""
        try:
            data = self.client.getDistanceSensorData(distance_sensor_name=sensor_name, vehicle_name=VEHICLE_NAME)
            if data and hasattr(data, 'distance') and data.distance > 0:
                return float(data.distance)
        except Exception:
            pass
        return None
    
    def get_all_distances(self):
        """Получает дистанции со всех distance сенсоров."""
        distances = {}
        for sensor in ["DistanceFront", "DistanceBack", "DistanceLeft", "DistanceRight"]:
            dist = self.get_distance_sensor(sensor)
            if dist is not None:
                distances[sensor] = dist
        return distances
    
    def get_obstacle_distance(self, direction=None):
        """
        Получает минимальную дистанцию до препятствия.
        Использует все доступные сенсоры: лидар, depth-камеру, distance сенсоры.
        direction: 'front', 'back', 'left', 'right' или None для минимума
        """
        distances = []
        
        # Лидар
        d_lidar = self.get_lidar_distance()
        if d_lidar is not None:
            distances.append(d_lidar)
        
        # Depth камера
        d_depth = self.get_depth_distance()
        if d_depth is not None:
            distances.append(d_depth)
        
        # Distance сенсоры
        all_dists = self.get_all_distances()
        if direction == 'front' and "DistanceFront" in all_dists:
            distances.append(all_dists["DistanceFront"])
        elif direction == 'back' and "DistanceBack" in all_dists:
            distances.append(all_dists["DistanceBack"])
        elif direction == 'left' and "DistanceLeft" in all_dists:
            distances.append(all_dists["DistanceLeft"])
        elif direction == 'right' and "DistanceRight" in all_dists:
            distances.append(all_dists["DistanceRight"])
        elif direction is None:
            # Используем все сенсоры
            distances.extend(all_dists.values())
        
        if not distances:
            return None
        
        return min(distances)
    
    def update_coordinates_display(self):
        """Обновляет отображение координат и скорости в симуляторе."""
        now = time.time()
        if now - self.last_coord_update < self.coord_update_interval:
            return
        
        self.last_coord_update = now
        
        if self.start_position is None:
            return
        
        x, y, z = self.get_position()
        sx, sy, sz = self.start_position
        
        # Координаты относительно стартовой точки
        rel_x = x - sx
        rel_y = y - sy
        rel_z = z - sz
        
        # Получаем текущую скорость (обновляем чаще для скорости)
        speed, vx, vy, vz = self.get_velocity()
        speed_kmh = speed * 3.6  # Конвертируем в км/ч
        
        # Display coordinates and speed in simulator (left side)
        # Format for better readability
        coord_text = f"X: {rel_x:6.2f}  Y: {rel_y:6.2f}  Z: {rel_z:6.2f}  |  Speed: {speed:5.2f} m/s ({speed_kmh:5.1f} km/h)"
        try:
            self.client.simPrintLogMessage("", coord_text, severity=0)
        except Exception:
            pass
    
    def update_speed_display(self):
        """Обновляет отображение скорости отдельно (для более частого обновления)."""
        now = time.time()
        if now - self.last_speed_display < self.speed_display_interval:
            return
        
        self.last_speed_display = now
        
        # Получаем текущую скорость
        speed, vx, vy, vz = self.get_velocity()
        speed_kmh = speed * 3.6
        
        # Display speed in left side of screen
        speed_text = f"Speed: {speed:.2f} m/s ({speed_kmh:.1f} km/h)  |  Vx: {vx:.2f}  Vy: {vy:.2f}  Vz: {vz:.2f}"
           
    def send_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """Отправляет команду скорости с контролем высоты (без блокировки)."""
        # Обновляем координаты и скорость на экране
        self.update_coordinates_display()
        self.update_speed_display()
        
        # Сохраняем точку маршрута периодически
        now = time.time()
        if now - self.last_log_time > 0.5:  # Каждые 0.5 секунды
            x, y, z = self.get_position()
            self.route_points.append((x, y, z))
            self.last_log_time = now
        
        # Собираем точки лидара для визуализации
        self.collect_lidar_points()
        
        # Проверяем текущую высоту и корректируем при необходимости
        current_x, current_y, current_z = self.get_position()
        height_error = current_z - FLIGHT_HEIGHT
        
        # Если дрон слишком низко (z больше, т.к. в NED отрицательное = вверх)
        if height_error > 0.3:  # Дрон ниже целевой высоты более чем на 0.3м
            # Корректируем вертикальную скорость для подъема
            vz = min(vz, -0.5)  # В NED отрицательное = вверх
        elif height_error < -0.3:  # Дрон выше целевой высоты более чем на 0.3м
            # Корректируем вертикальную скорость для снижения
            vz = max(vz, 0.3)  # В NED положительное = вниз
        else:
            # Поддерживаем высоту - небольшая коррекция
            vz_correction = -height_error * 0.5  # Пропорциональная коррекция
            vz = vz + vz_correction
        
        # Ограничиваем вертикальную скорость
        vz = np.clip(vz, -1.0, 1.0)
        
        # Используем moveByVelocityAsync с контролем yaw для предотвращения вращения
        # yaw_mode устанавливаем для предотвращения автоматического поворота
        self.client.moveByVelocityAsync(
            float(vx), float(vy), float(vz),
            duration=CMD_DT,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0),  # Не вращаемся автоматически
            vehicle_name=VEHICLE_NAME
        )
        time.sleep(CMD_DT * 0.9)
    
    def hover(self):
        """Переход в режим зависания."""
        self.client.hoverAsync(vehicle_name=VEHICLE_NAME)
        time.sleep(0.2)
    
    def move_to_safe(self, target_x, target_y, target_z):
        """
        Безопасное перемещение к точке с обходом препятствий.
        Возвращает True если достигли точки, False если не удалось.
        """
        target_z = float(target_z)
        
        avoid_count = 0
        side_direction = 1  # Направление обхода: 1 = влево, -1 = вправо
        
        while True:
            # Проверка столкновения
            if self.has_collision():
                log_warn("Столкновение! Аварийный подъём...")
                self.hover()
                # Аварийный подъём
                for _ in range(20):
                    self.send_velocity(0, 0, -AVOID_VERTICAL_SPEED)
                return False
            
            # Текущая позиция
            x, y, z = self.get_position()
            
            # Расстояние до цели
            dx = target_x - x
            dy = target_y - y
            dz = target_z - z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Достигли цели?
            if dist < ARRIVAL_TOL:
                self.hover()
                return True
            
            # Единичный вектор к цели
            ux, uy, uz = dx/dist, dy/dist, dz/dist
            
            # Логируем позицию периодически
            now = time.time()
            if now - self.last_log_time > self.log_interval:
                print(f"  Позиция: ({x:.1f}, {y:.1f}, {z:.1f}) -> цель: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
                self.last_log_time = now
            
            # Проверяем препятствия
            obstacle_dist = self.get_obstacle_distance()
            
            if obstacle_dist is not None:
                
                # АВАРИЙНАЯ СИТУАЦИЯ: очень близко
                if obstacle_dist < STOP_DISTANCE:
                    avoid_count += 1
                    if avoid_count > MAX_AVOID_ITERATIONS:
                        log_warn(f"Не удалось обойти препятствие, пропускаем точку")
                        self.hover()
                        return False
                    
                    print(f"  АВАРИЙНО БЛИЗКО ({obstacle_dist:.1f}м)! Подъём...")
                    # Резкий подъём
                    for _ in range(15):
                        self.send_velocity(0, 0, -AVOID_VERTICAL_SPEED * 1.5)
                    continue
                
                # ОПАСНО: начинаем обход
                if obstacle_dist < WARN_DISTANCE:
                    avoid_count += 1
                    if avoid_count > MAX_AVOID_ITERATIONS:
                        log_warn(f"Слишком долго обходим, пропускаем точку")
                        self.hover()
                        return False
                    
                    # Вектор влево от направления движения
                    left_x, left_y = -uy, ux
                    
                    # Уходим в сторону + немного вверх
                    vx = left_x * AVOID_LATERAL_SPEED * side_direction
                    vy = left_y * AVOID_LATERAL_SPEED * side_direction
                    vz = -AVOID_VERTICAL_SPEED * 0.5  # Немного вверх (NED)
                    
                    print(f"  Обход препятствия ({obstacle_dist:.1f}м), сторона: {'лево' if side_direction > 0 else 'право'}")
                    self.send_velocity(vx, vy, vz)
                    
                    # Периодически меняем направление обхода
                    if avoid_count % 8 == 0:
                        side_direction *= -1
                    
                    continue
                
                # ПРЕДУПРЕЖДЕНИЕ: замедляемся
                if obstacle_dist < SAFE_DISTANCE:
                    # Коэффициент замедления
                    slowdown = (obstacle_dist - WARN_DISTANCE) / (SAFE_DISTANCE - WARN_DISTANCE)
                    slowdown = max(0.3, min(1.0, slowdown))
                    speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * slowdown
                else:
                    speed = MAX_SPEED
                    avoid_count = 0  # Сбрасываем счётчик обхода
            else:
                # Сенсоры недоступны - летим медленно
                speed = MIN_SPEED
            
            # Вычисляем скорость к цели
            speed = min(speed, dist)  # Замедляемся при подлёте
            speed = max(MIN_SPEED, speed)
            
            vx = ux * speed
            vy = uy * speed
            
            # Вертикальная скорость: приоритет поддержанию высоты
            # Если горизонтальное движение, вертикальная скорость минимальна
            if abs(dx) > 0.1 or abs(dy) > 0.1:  # Есть горизонтальное движение
                # При горизонтальном движении стараемся поддерживать высоту
                height_error = z - target_z
                if abs(height_error) > 0.2:  # Нужна коррекция высоты
                    vz = -height_error * 0.8  # Пропорциональная коррекция (NED: отрицательное = вверх)
                    vz = np.clip(vz, -0.8, 0.8)
                else:
                    vz = 0.0  # Поддерживаем высоту
            else:
                # Вертикальное движение к цели
                vz = np.clip(uz * speed, -0.8, 0.8)  # Ограничиваем вертикальную скорость
            
            self.send_velocity(vx, vy, vz)


# =========================
# ОПРЕДЕЛЕНИЕ ПОЗИЦИИ И ПОИСК СТЕН
# =========================

def find_nearest_wall(drone: SafeDrone):
    """
    Определяет направление к ближайшей стене используя все сенсоры.
    Возвращает (direction, distance), где direction: 'front', 'back', 'left', 'right'
    """
    distances = {}
    
    # Получаем расстояния со всех сенсоров
    for sensor_name, direction in [("DistanceFront", "front"), 
                                   ("DistanceBack", "back"),
                                   ("DistanceLeft", "left"),
                                   ("DistanceRight", "right")]:
        try:
            dist = drone.get_distance_sensor(sensor_name)
            if dist is not None and dist < 50.0:  # Разумный предел
                distances[direction] = dist
        except Exception:
            pass
    
    # Также проверяем лидар и depth камеру для направления вперед
    lidar_dist = drone.get_lidar_distance()
    depth_dist = drone.get_depth_distance()
    
    if lidar_dist is not None:
        if "front" not in distances or lidar_dist < distances["front"]:
            distances["front"] = lidar_dist
    
    if depth_dist is not None:
        if "front" not in distances or depth_dist < distances["front"]:
            distances["front"] = depth_dist
    
    if not distances:
        return None, None
    
    # Находим направление с минимальным расстоянием
    nearest_direction = min(distances, key=distances.get)
    nearest_distance = distances[nearest_direction]
    
    return nearest_direction, nearest_distance

def get_wall_direction_vector(direction):
    """
    Возвращает единичный вектор направления к стене.
    direction: 'front', 'back', 'left', 'right'
    """
    vectors = {
        'front': (1.0, 0.0),   # Вперед по X
        'back': (-1.0, 0.0),   # Назад по X
        'left': (0.0, 1.0),     # Влево по Y
        'right': (0.0, -1.0)    # Вправо по Y
    }
    return vectors.get(direction, (1.0, 0.0))

def fly_to_wall(drone: SafeDrone, target_distance=PERIMETER_OFFSET):
    """
    Летит к ближайшей стене до заданного расстояния.
    Возвращает True если достигли стены, False если не удалось.
    """
    log_info("Поиск ближайшей стены...")
    
    max_attempts = 100
    attempt = 0
    
    while attempt < max_attempts:
        direction, distance = find_nearest_wall(drone)
        
        if direction is None or distance is None:
            log_warn("Не удалось определить направление к стене")
            time.sleep(0.5)
            attempt += 1
            continue
        
        log_info(f"Ближайшая стена: {direction}, расстояние: {distance:.2f} м")
        
        # Если уже на нужном расстоянии
        if distance <= target_distance + 0.5:
            log_ok(f"Достигнута стена на расстоянии {distance:.2f} м")
            return True, direction
        
        # Вычисляем вектор к стене
        vx, vy = get_wall_direction_vector(direction)
        
        # Летим к стене
        x, y, z = drone.get_position()
        target_x = x + vx * (distance - target_distance) * 0.5
        target_y = y + vy * (distance - target_distance) * 0.5
        
        # Перемещаемся небольшими шагами
        reached = drone.move_to_safe(target_x, target_y, FLIGHT_HEIGHT)
        
        if not reached:
            log_warn("Не удалось приблизиться к стене, пробуем снова...")
        
        attempt += 1
        time.sleep(0.1)
    
    log_warn("Не удалось достичь стены за отведенное время")
    return False, None

def fly_to_wall_in_front(drone: SafeDrone, target_distance=PERIMETER_OFFSET):
    """
    Летит к стене прямо перед дроном до заданного расстояния.
    Возвращает True если достигли стены, False если не удалось.
    """
    log_info("Поиск стены впереди...")
    
    max_iterations = 500
    iteration = 0
    last_log_time = 0
    log_interval = 2.0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Проверяем расстояние до стены впереди
        front_dist = drone.get_obstacle_distance('front')
        
        if front_dist is None:
            # Если сенсоры не дают данных, летим медленно вперед
            if time.time() - last_log_time > log_interval:
                log_warn("Сенсоры не дают данных, летим медленно вперед...")
                last_log_time = time.time()
            # Летим вперед медленно
            drone.send_velocity(MIN_SPEED, 0, 0)
            time.sleep(CMD_DT)
            continue
        
        # Проверяем высоту и корректируем при необходимости
        x, y, z = drone.get_position()
        height_error = z - FLIGHT_HEIGHT
        
        # Логируем периодически
        if time.time() - last_log_time > log_interval:
            log_info(f"Расстояние до стены: {front_dist:.2f} м (цель: {target_distance:.2f} м), высота: {z:.2f} м (цель: {FLIGHT_HEIGHT:.2f} м)")
            last_log_time = time.time()
        
        # Если дрон слишком низко, сначала поднимаемся
        if height_error > 0.5:  # Дрон ниже целевой высоты более чем на 0.5м
            log_warn(f"Дрон слишком низко ({z:.2f} м), поднимаемся...")
            drone.send_velocity(0, 0, -0.8)  # Подъем (NED: отрицательное = вверх)
            time.sleep(CMD_DT)
            continue
        
        # Если уже на нужном расстоянии (с допуском)
        if abs(front_dist - target_distance) < 0.5:
            log_ok(f"Достигнута стена на расстоянии {front_dist:.2f} м (цель: {target_distance:.2f} м)")
            drone.hover()
            time.sleep(0.5)
            return True
        
        # Вычисляем скорость движения
        if front_dist > target_distance:
            # Слишком далеко, приближаемся
            # Замедляемся при приближении к цели
            distance_to_target = front_dist - target_distance
            if distance_to_target < 2.0:
                # Близко к цели - медленно
                speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (distance_to_target / 2.0)
                speed = max(MIN_SPEED, min(MAX_SPEED, speed))
            else:
                # Далеко - нормальная скорость
                speed = MAX_SPEED
            
            # Летим вперед, поддерживая высоту (vz=0, высота контролируется в send_velocity)
            drone.send_velocity(speed, 0, 0)
        else:
            # Слишком близко к стене, отдаляемся назад, поддерживая высоту
            distance_to_target = target_distance - front_dist
            speed = MIN_SPEED * 0.5  # Медленно назад
            drone.send_velocity(-speed, 0, 0)  # Высота контролируется в send_velocity
        
        time.sleep(CMD_DT)
    
    log_warn(f"Не удалось достичь стены за {max_iterations} итераций")
    drone.hover()
    return False


def fly_perimeter_4_walls(drone: SafeDrone):
    """
    Полет по периметру: находит 4 стены, летит к каждой, поворачивается на 90° вправо.
    """
    log_info("Начинаем полет по периметру (4 стены)...")
    
    walls_found = []
    
    for wall_num in range(1, 5):
        log_info(f"=== СТЕНА {wall_num}/4 ===")
        
        # 1. Ищем стену впереди
        log_info("Поиск стены впереди...")
        reached = fly_to_wall_in_front(drone, target_distance=PERIMETER_OFFSET)
        
        if not reached:
            log_warn(f"Не удалось достичь стены {wall_num}, пропускаем...")
            continue
        
        # Сохраняем позицию стены
        wall_pos = drone.get_position()
        walls_found.append(wall_pos)
        log_ok(f"Стена {wall_num} найдена на позиции: ({wall_pos[0]:.2f}, {wall_pos[1]:.2f}, {wall_pos[2]:.2f})")
        
        # Небольшая пауза для стабилизации
        time.sleep(0.5)
        
        # 2. Поворачиваемся на 90° вправо (кроме последней стены)
        if wall_num < 4:
            log_info(f"Поворот на 90° вправо...")
            drone.rotate_right_90()
            
            # Пауза для стабилизации после поворота
            time.sleep(1.0)
            
            # Проверяем, видим ли мы следующую стену
            front_dist = drone.get_obstacle_distance('front')
            if front_dist is not None:
                log_info(f"После поворота расстояние до следующей стены: {front_dist:.2f} м")
            else:
                log_warn("После поворота не видно стены впереди, продолжаем...")
        else:
            log_info("Последняя стена найдена, поворот не требуется")
        
        print()
    
    log_ok(f"Найдено стен: {len(walls_found)}/4")
    return len(walls_found) == 4


def fly_perimeter_auto_wall_detection(drone: SafeDrone):
    """
    Автоматический облет периметра: дрон летит вперед, при остановке перед стеной
    поворачивает направо на 90° и продолжает полет до облета всех стен.
    """
    log_info("Начинаем автоматический облет периметра с обнаружением стен...")
    
    # Параметры обнаружения остановки
    STOP_SPEED_THRESHOLD = 0.3  # м/с - скорость считается остановкой
    WALL_DETECTION_DISTANCE = 3.0  # м - расстояние до стены для обнаружения
    STUCK_CHECK_INTERVAL = 1.0  # секунды - интервал проверки остановки
    MIN_STUCK_TIME = 2.0  # секунды - минимальное время остановки для поворота
    
    # Параметры облета
    MAX_TURNS = 20  # максимальное количество поворотов (защита от бесконечного цикла)
    RETURN_TO_START_DISTANCE = 5.0  # м - расстояние до старта для определения завершения
    
    start_pos = drone.get_position()
    start_x, start_y, start_z = start_pos
    
    turn_count = 0
    last_position = start_pos
    stuck_start_time = None
    last_stuck_check = 0
    last_log_time = 0
    log_interval = 3.0
    
    log_info(f"Стартовая позиция: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f})")
    log_info("Дрон будет лететь вперед и автоматически поворачивать направо при обнаружении стены")
    
    iteration = 0
    max_iterations = 10000  # защита от бесконечного цикла
    
    while iteration < max_iterations:
        iteration += 1
        
        # Получаем текущую позицию и скорость
        current_pos = drone.get_position()
        x, y, z = current_pos
        speed, vx, vy, vz = drone.get_velocity()
        
        # Проверяем расстояние до стены впереди
        front_dist = drone.get_obstacle_distance('front')
        
        # Логируем периодически
        now = time.time()
        if now - last_log_time > log_interval:
            log_info(f"Позиция: ({x:.2f}, {y:.2f}, {z:.2f}) | Скорость: {speed:.2f} м/с | "
                    f"Расстояние до стены: {front_dist:.2f} м" if front_dist else "N/A")
            last_log_time = now
        
        # Проверяем, вернулись ли мы в стартовую точку (завершение облета)
        dist_to_start = math.sqrt((x - start_x)**2 + (y - start_y)**2)
        if dist_to_start < RETURN_TO_START_DISTANCE and turn_count >= 4:
            log_ok(f"Облет периметра завершен! Вернулись в стартовую точку (расстояние: {dist_to_start:.2f} м)")
            log_ok(f"Всего выполнено поворотов: {turn_count}")
            drone.hover()
            return True
        
        # Проверяем остановку перед стеной
        if now - last_stuck_check >= STUCK_CHECK_INTERVAL:
            last_stuck_check = now
            
            # Проверяем скорость - дрон остановился?
            is_stopped = speed < STOP_SPEED_THRESHOLD
            
            # Проверяем расстояние до стены - стена близко?
            wall_close = front_dist is not None and front_dist < WALL_DETECTION_DISTANCE
            
            # Проверяем, не движется ли дрон (позиция не меняется)
            pos_change = math.sqrt((x - last_position[0])**2 + (y - last_position[1])**2)
            not_moving = pos_change < 0.5  # менее 0.5 м за интервал проверки
            
            if is_stopped and wall_close and not_moving:
                # Дрон остановился перед стеной
                if stuck_start_time is None:
                    stuck_start_time = now
                    log_warn(f"Обнаружена остановка перед стеной (расстояние: {front_dist:.2f} м, скорость: {speed:.2f} м/с)")
                else:
                    # Проверяем, достаточно ли долго дрон стоит
                    stuck_duration = now - stuck_start_time
                    if stuck_duration >= MIN_STUCK_TIME:
                        if turn_count >= MAX_TURNS:
                            log_warn(f"Достигнуто максимальное количество поворотов ({MAX_TURNS}), завершаем облет")
                            drone.hover()
                            return False
                        
                        # Поворачиваем направо на 90°
                        log_info(f"Дрон остановился перед стеной на {stuck_duration:.1f} сек. Поворот направо на 90°...")
                        drone.hover()
                        time.sleep(0.5)
                        
                        success = drone.rotate_right_90()
                        if success:
                            turn_count += 1
                            log_ok(f"Поворот {turn_count} выполнен. Продолжаем полет...")
                            
                            # Сбрасываем таймер остановки
                            stuck_start_time = None
                            last_position = current_pos
                            
                            # Пауза после поворота
                            time.sleep(0.5)
                        else:
                            log_warn("Не удалось выполнить поворот, продолжаем попытки...")
                            stuck_start_time = None  # Сбрасываем, чтобы попробовать снова
            else:
                # Дрон движется или стена далеко - сбрасываем таймер остановки
                if stuck_start_time is not None:
                    stuck_start_time = None
        
        # Если дрон не остановился, продолжаем лететь вперед
        if stuck_start_time is None:
            # Вычисляем скорость движения вперед
            if front_dist is None:
                # Сенсоры не дают данных - летим медленно
                forward_speed = MIN_SPEED
            elif front_dist < STOP_DISTANCE:
                # Очень близко к стене - останавливаемся
                forward_speed = 0.0
                drone.hover()
            elif front_dist < WARN_DISTANCE:
                # Близко к стене - замедляемся
                forward_speed = MIN_SPEED * 0.5
            elif front_dist < SAFE_DISTANCE:
                # Приближаемся к стене - начинаем замедление
                slowdown_factor = (front_dist - WARN_DISTANCE) / (SAFE_DISTANCE - WARN_DISTANCE)
                forward_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * slowdown_factor
            else:
                # Нет препятствий - нормальная скорость
                forward_speed = MAX_SPEED
            
            # Летим вперед (в направлении текущего yaw)
            # Для упрощения используем скорость по X (вперед в локальной системе координат)
            # В AirSim при yaw=0, движение по X = вперед
            drone.send_velocity(forward_speed, 0, 0)
        
        # Обновляем последнюю позицию
        last_position = current_pos
        
        # Небольшая пауза
        time.sleep(CMD_DT)
    
    log_warn(f"Достигнуто максимальное количество итераций ({max_iterations})")
    drone.hover()
    return False


def fly_along_wall(drone: SafeDrone, wall_direction, distance_to_fly=100.0):
    """
    Летит вдоль стены в заданном направлении, поддерживая постоянное расстояние до стены.
    wall_direction: направление к стене ('front', 'back', 'left', 'right')
    distance_to_fly: расстояние для полета вдоль стены (метры)
    """
    log_info(f"Начинаем полет вдоль стены (направление к стене: {wall_direction})")
    
    # Определяем направление движения вдоль стены (по часовой стрелке)
    # Если стена спереди, летим вправо
    # Если стена справа, летим назад
    # Если стена сзади, летим влево
    # Если стена слева, летим вперед
    along_directions = {
        'front': (0.0, -1.0),   # Вправо (отрицательный Y)
        'right': (-1.0, 0.0),    # Назад (отрицательный X)
        'back': (0.0, 1.0),      # Влево (положительный Y)
        'left': (1.0, 0.0)       # Вперед (положительный X)
    }
    
    along_vx, along_vy = along_directions.get(wall_direction, (0.0, -1.0))
    
    start_x, start_y, _ = drone.get_position()
    start_distance = 0.0
    max_iterations = int(distance_to_fly / MAX_SPEED * CMD_HZ) + 500
    iteration = 0
    last_correction_time = 0
    correction_cooldown = 1.0  # Минимальный интервал между коррекциями (секунды)
    
    log_info(f"Направление движения вдоль стены: ({along_vx:.1f}, {along_vy:.1f})")
    log_info(f"Целевое расстояние до стены: {PERIMETER_OFFSET:.1f} м")
    
    while start_distance < distance_to_fly and iteration < max_iterations:
        iteration += 1
        
        # Получаем текущую позицию
        x, y, z = drone.get_position()
        
        # Проверяем расстояние до стены в нужном направлении
        direction, wall_dist = find_nearest_wall(drone)
        
        # Если стена не в ожидаемом направлении, пытаемся найти ее снова
        if direction != wall_direction:
            # Проверяем, может быть мы повернули за угол
            log_info(f"Направление стены изменилось: {direction} (было {wall_direction})")
            # Обновляем направление к стене
            wall_direction = direction
            along_vx, along_vy = along_directions.get(wall_direction, (0.0, -1.0))
            log_info(f"Обновлено направление движения: ({along_vx:.1f}, {along_vy:.1f})")
        
        # Вычисляем скорость движения вдоль стены
        vx_along = along_vx * MAX_SPEED
        vy_along = along_vy * MAX_SPEED
        
        # Корректируем расстояние до стены (только если прошло достаточно времени)
        now = time.time()
        correction_vx = 0.0
        correction_vy = 0.0
        
        if wall_dist is not None and (now - last_correction_time) > correction_cooldown:
            vx_wall, vy_wall = get_wall_direction_vector(wall_direction)
            
            if wall_dist > PERIMETER_OFFSET + 1.0:
                # Слишком далеко от стены, приближаемся медленно
                correction_vx = vx_wall * MIN_SPEED * 0.3
                correction_vy = vy_wall * MIN_SPEED * 0.3
                last_correction_time = now
            elif wall_dist < PERIMETER_OFFSET - 1.0:
                # Слишком близко к стене, отдаляемся медленно
                correction_vx = -vx_wall * MIN_SPEED * 0.3
                correction_vy = -vy_wall * MIN_SPEED * 0.3
                last_correction_time = now
        
        # Комбинируем скорости: движение вдоль стены + коррекция расстояния
        final_vx = vx_along + correction_vx
        final_vy = vy_along + correction_vy
        
        # Ограничиваем общую скорость
        speed = math.sqrt(final_vx**2 + final_vy**2)
        if speed > MAX_SPEED:
            final_vx = final_vx * MAX_SPEED / speed
            final_vy = final_vy * MAX_SPEED / speed
        
        # Отправляем команду скорости напрямую (без move_to_safe, чтобы избежать вращения)
        drone.send_velocity(final_vx, final_vy, 0.0)
        
        # Вычисляем пройденное расстояние от стартовой точки
        new_distance = math.sqrt((x - start_x)**2 + (y - start_y)**2)
        start_distance = new_distance
        
        # Логируем прогресс периодически
        if iteration % 50 == 0:
            log_info(f"Пройдено: {start_distance:.1f} м / {distance_to_fly:.1f} м, расстояние до стены: {wall_dist:.2f} м")
        
        if start_distance >= distance_to_fly:
            log_ok(f"Пройдено {start_distance:.2f} м вдоль стены")
            break
        
        # Небольшая пауза
        time.sleep(CMD_DT)
    
    if iteration >= max_iterations:
        log_warn(f"Достигнуто максимальное количество итераций ({max_iterations})")
    
    log_ok("Полет вдоль стены завершен")
    return True

# =========================
# ПОЛЕТ ПО ПЕРИМЕТРУ
# =========================

def calculate_perimeter_waypoints(start_x, start_y, size_x, size_y, offset):
    """
    Вычисляет точки маршрута по периметру по часовой стрелке.
    Возвращает список точек (x, y) для облета периметра.
    """
    # Вычисляем границы периметра
    min_x = start_x - size_x / 2 + offset
    max_x = start_x + size_x / 2 - offset
    min_y = start_y - size_y / 2 + offset
    max_y = start_y + size_y / 2 - offset
    
    waypoints = []
    
    # Полет по часовой стрелке:
    # 1. Верхний левый угол -> верхний правый угол
    waypoints.append((min_x, min_y))
    waypoints.append((max_x, min_y))
    
    # 2. Верхний правый угол -> нижний правый угол
    waypoints.append((max_x, max_y))
    
    # 3. Нижний правый угол -> нижний левый угол
    waypoints.append((min_x, max_y))
    
    # 4. Нижний левый угол -> верхний левый угол (замыкаем периметр)
    waypoints.append((min_x, min_y))
    
    return waypoints

def fly_perimeter(drone: SafeDrone):
    """Выполняет полет по периметру карты по часовой стрелке."""
    
    if drone.start_position is None:
        log_error("Стартовая позиция не установлена!")
        return False
    
    start_x, start_y, start_z = drone.start_position
    
    log_info("Параметры полета по периметру:")
    print(f"  Размер области: {PERIMETER_SIZE_X}x{PERIMETER_SIZE_Y} м")
    print(f"  Высота полета: {-FLIGHT_HEIGHT:.1f} м")
    print(f"  Скорость: {SPEED_KMH} км/ч ({SPEED_MS:.2f} м/с)")
    print(f"  Отступ от края: {PERIMETER_OFFSET} м")
    print(f"  Стартовая позиция: ({start_x:.1f}, {start_y:.1f}, {start_z:.1f})")
    
    # Вычисляем точки маршрута
    waypoints = calculate_perimeter_waypoints(
        start_x, start_y, 
        PERIMETER_SIZE_X, PERIMETER_SIZE_Y, 
        PERIMETER_OFFSET
    )
    
    log_info(f"Маршрут состоит из {len(waypoints)} точек")
    
    # Летим по периметру
    for i, (wx, wy) in enumerate(waypoints):
        log_info(f"=== Точка {i+1}/{len(waypoints)}: ({wx:.1f}, {wy:.1f}) ===")
        
        # Перемещаемся к точке на заданной высоте
        reached = drone.move_to_safe(wx, wy, FLIGHT_HEIGHT)
        
        if reached:
            log_ok(f"Достигнута точка {i+1}")
        else:
            log_warn(f"Не удалось достичь точки {i+1}, продолжаем...")
        
        # Небольшая пауза для стабилизации
        time.sleep(0.5)
    
    log_ok("Полет по периметру завершен")
    return True


# =========================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =========================

def save_route_results(drone: SafeDrone):
    """Сохраняет результаты полета по периметру и создает визуализацию маршрута."""
    
    if not drone.route_points:
        log_warn("Нет данных маршрута для сохранения")
        return None, None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "maps")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join(output_dir, f"perimeter_route_{timestamp}")
    
    # Сохраняем JSON
    json_path = f"{base_path}.json"
    
    # Вычисляем статистику маршрута
    route_length = 0.0
    for i in range(1, len(drone.route_points)):
        x1, y1, z1 = drone.route_points[i-1]
        x2, y2, z2 = drone.route_points[i]
        route_length += math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    # Преобразуем точки маршрута относительно стартовой позиции
    if drone.start_position:
        sx, sy, sz = drone.start_position
        relative_route = [(x-sx, y-sy, z-sz) for x, y, z in drone.route_points]
    else:
        relative_route = drone.route_points
    
    data = {
        "flight_parameters": {
            "perimeter_size_x": PERIMETER_SIZE_X,
            "perimeter_size_y": PERIMETER_SIZE_Y,
            "flight_height": -FLIGHT_HEIGHT,
            "speed_kmh": SPEED_KMH,
            "speed_ms": SPEED_MS,
            "perimeter_offset": PERIMETER_OFFSET,
            "timestamp": timestamp
        },
        "start_position": drone.start_position,
        "route_points": drone.route_points,
        "relative_route": relative_route,
        "lidar_points": drone.lidar_points if drone.lidar_points else [],
        "statistics": {
            "total_points": len(drone.route_points),
            "lidar_points_count": len(drone.lidar_points) if drone.lidar_points else 0,
            "route_length_m": route_length,
            "estimated_duration_s": route_length / SPEED_MS if SPEED_MS > 0 else 0
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    log_ok(f"JSON: {json_path}")
    
    # Создаём визуализацию маршрута (как у робота-пылесоса)
    log_info("Создание визуализации маршрута...")
    
    if not drone.route_points:
        log_warn("Нет точек маршрута для визуализации")
        return json_path, None
    
    # Извлекаем координаты маршрута
    route_x = [p[0] for p in drone.route_points]
    route_y = [p[1] for p in drone.route_points]
    route_z = [p[2] for p in drone.route_points]
    
    # Вычисляем относительные координаты
    if drone.start_position:
        sx, sy, sz = drone.start_position
        rel_x = [x - sx for x in route_x]
        rel_y = [y - sy for y in route_y]
        rel_z = [z - sz for z in route_z]
    else:
        rel_x = route_x
        rel_y = route_y
        rel_z = route_z
    
    # Обрабатываем точки лидара
    lidar_rel_x = []
    lidar_rel_y = []
    lidar_rel_z = []
    
    if drone.lidar_points and drone.start_position:
        sx, sy, sz = drone.start_position
        for pt in drone.lidar_points:
            if len(pt) >= 3:
                lidar_rel_x.append(pt[0] - sx)
                lidar_rel_y.append(pt[1] - sy)
                lidar_rel_z.append(pt[2] - sz)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Вид сверху - карта маршрута с точками лидара
    ax1 = plt.subplot(2, 3, 1)
    
    # Отображаем точки лидара (если есть)
    if lidar_rel_x:
        # Подвыборка для производительности (каждая 5-я точка)
        lidar_sample_x = lidar_rel_x[::5]
        lidar_sample_y = lidar_rel_y[::5]
        ax1.scatter(lidar_sample_x, lidar_sample_y, c='gray', s=0.5, alpha=0.3, label=f'Точки лидара ({len(lidar_rel_x)})')
    
    ax1.plot(rel_x, rel_y, 'b-', linewidth=2, alpha=0.7, label='Маршрут')
    ax1.plot(rel_x[0], rel_y[0], 'go', markersize=10, label='Старт')
    ax1.plot(rel_x[-1], rel_y[-1], 'ro', markersize=10, label='Финиш')
    
    # Добавляем стрелки для направления движения
    if len(rel_x) > 10:
        step = len(rel_x) // 20
        for i in range(0, len(rel_x) - step, step):
            dx = rel_x[i+step] - rel_x[i]
            dy = rel_y[i+step] - rel_y[i]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                ax1.arrow(rel_x[i], rel_y[i], dx*0.3, dy*0.3, 
                         head_width=0.5, head_length=0.5, fc='blue', ec='blue', alpha=0.5)
    
    # Показываем периметр области
    if drone.start_position:
        min_x = -PERIMETER_SIZE_X / 2 + PERIMETER_OFFSET
        max_x = PERIMETER_SIZE_X / 2 - PERIMETER_OFFSET
        min_y = -PERIMETER_SIZE_Y / 2 + PERIMETER_OFFSET
        max_y = PERIMETER_SIZE_Y / 2 - PERIMETER_OFFSET
        perimeter_x = [min_x, max_x, max_x, min_x, min_x]
        perimeter_y = [min_y, min_y, max_y, max_y, min_y]
        ax1.plot(perimeter_x, perimeter_y, 'r--', linewidth=2, alpha=0.5, label='Целевой периметр')
    
    ax1.set_xlabel("X относительно старта (м)")
    ax1.set_ylabel("Y относительно старта (м)")
    ax1.set_title("Карта маршрута с точками лидара (вид сверху)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # График высоты по времени/расстоянию
    ax2 = plt.subplot(2, 3, 2)
    distances = [0]
    cum_dist = 0
    for i in range(1, len(rel_x)):
        dist = math.sqrt((rel_x[i]-rel_x[i-1])**2 + (rel_y[i]-rel_y[i-1])**2)
        cum_dist += dist
        distances.append(cum_dist)
    
    ax2.plot(distances, rel_z, 'g-', linewidth=2)
    ax2.axhline(y=rel_z[0], color='r', linestyle='--', alpha=0.5, label=f'Целевая высота: {-FLIGHT_HEIGHT:.1f} м')
    ax2.set_xlabel("Пройденное расстояние (м)")
    ax2.set_ylabel("Высота относительно старта (м)")
    ax2.set_title("Высота полета")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3D визуализация маршрута с точками лидара
    ax3 = plt.subplot(2, 3, 3, projection="3d")
    if lidar_rel_x:
        lidar_sample_x = lidar_rel_x[::10]
        lidar_sample_y = lidar_rel_y[::10]
        lidar_sample_z = lidar_rel_z[::10]
        ax3.scatter(lidar_sample_x, lidar_sample_y, lidar_sample_z, 
                   c='gray', s=0.5, alpha=0.2, label=f'Точки лидара')
    ax3.plot(rel_x, rel_y, rel_z, 'b-', linewidth=2, alpha=0.7, label='Маршрут')
    ax3.scatter(rel_x[0], rel_y[0], rel_z[0], c='green', s=100, label='Старт')
    ax3.scatter(rel_x[-1], rel_y[-1], rel_z[-1], c='red', s=100, label='Финиш')
    ax3.set_xlabel("X (м)")
    ax3.set_ylabel("Y (м)")
    ax3.set_zlabel("Z (м)")
    ax3.set_title("3D маршрут с точками лидара")
    ax3.legend()
    
    # Визуализация точек лидара (вид сбоку)
    ax4 = plt.subplot(2, 3, 4)
    if lidar_rel_x:
        lidar_sample_x = lidar_rel_x[::5]
        lidar_sample_z = lidar_rel_z[::5]
        ax4.scatter(lidar_sample_x, lidar_sample_z, c='gray', s=0.5, alpha=0.3, label='Точки лидара')
    ax4.plot(rel_x, rel_z, 'b-', linewidth=2, alpha=0.7, label='Маршрут')
    ax4.set_xlabel("X относительно старта (м)")
    ax4.set_ylabel("Z (высота) относительно старта (м)")
    ax4.set_title("Вид сбоку: маршрут и точки лидара")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Визуализация точек лидара (вид спереди)
    ax5 = plt.subplot(2, 3, 5)
    if lidar_rel_y:
        lidar_sample_y = lidar_rel_y[::5]
        lidar_sample_z = lidar_rel_z[::5]
        ax5.scatter(lidar_sample_y, lidar_sample_z, c='gray', s=0.5, alpha=0.3, label='Точки лидара')
    ax5.plot(rel_y, rel_z, 'b-', linewidth=2, alpha=0.7, label='Маршрут')
    ax5.set_xlabel("Y относительно старта (м)")
    ax5.set_ylabel("Z (высота) относительно старта (м)")
    ax5.set_title("Вид спереди: маршрут и точки лидара")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Статистика
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    lidar_count = len(drone.lidar_points) if drone.lidar_points else 0
    stats_text = f"""
СТАТИСТИКА ПОЛЕТА

Всего точек маршрута:  {len(drone.route_points):,}
Точек лидара собрано:  {lidar_count:,}
Длина маршрута:        {route_length:.2f} м
Скорость полета:       {SPEED_KMH} км/ч ({SPEED_MS:.2f} м/с)
Высота полета:        {-FLIGHT_HEIGHT:.1f} м
Размер области:        {PERIMETER_SIZE_X}x{PERIMETER_SIZE_Y} м

Стартовая позиция:
  X: {drone.start_position[0]:.2f} м
  Y: {drone.start_position[1]:.2f} м
  Z: {drone.start_position[2]:.2f} м

Время: {timestamp}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family="monospace", va="center")
    
    plt.tight_layout()
    
    png_path = f"{base_path}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    log_ok(f"PNG: {png_path}")
    
    return json_path, png_path


# =========================
# ГЛАВНАЯ ФУНКЦИЯ
# =========================

def main():
    print("=" * 60)
    print("  ПОЛЕТ КВАДРОКОПТЕРА ПО ПЕРИМЕТРУ КАРТЫ")
    print("=" * 60)
    print()
    
    # 1. Проверяем и восстанавливаем settings.json (один раз!)
    if not ensure_settings():
        log_error("Не удалось настроить settings.json!")
        log_info("Создайте файл вручную или проверьте права доступа")
        sys.exit(1)
    
    print()
    
    # 2. Ожидаем симулятор (пытаемся запустить автоматически)
    wait_for_simulator(timeout=90)  # Не критично, если не обнаружен - попробуем подключиться
    
    print()
    
    # 3. Подключаемся к AirSim (с множественными попытками)
    log_info("Попытка подключения к AirSim...")
    client = connect_to_airsim(max_retries=40, retry_delay=2.0)
    if client is None:
        log_error("Не удалось подключиться к AirSim!")
        sys.exit(1)
    
    print()
    
    # 4. Создаём безопасный дрон
    drone = SafeDrone(client)
    
    try:
        # 5. Инициализация и взлёт
        drone.initialize()
        drone.takeoff()
        
        # Небольшая пауза для стабилизации после взлета
        log_info("Стабилизация после взлета...")
        time.sleep(2.0)
        
        # Поднимаемся на рабочую высоту
        log_info(f"Подъем на рабочую высоту {-FLIGHT_HEIGHT:.1f} м...")
        drone.move_to_safe(drone.start_position[0], drone.start_position[1], FLIGHT_HEIGHT)
        
        # Небольшая пауза для стабилизации на рабочей высоте
        time.sleep(1.0)
        
        # Определяем текущую позицию
        x, y, z = drone.get_position()
        log_info(f"Текущая позиция дрона: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        log_info(f"Относительно старта: X={x-drone.start_position[0]:.2f}, Y={y-drone.start_position[1]:.2f}, Z={z-drone.start_position[2]:.2f}")
        
        # Показываем информацию о сенсорах
        log_info("Проверка сенсоров для определения стен...")
        all_dists = drone.get_all_distances()
        if all_dists:
            for sensor, dist in all_dists.items():
                log_info(f"  {sensor}: {dist:.2f} м")
        else:
            log_warn("Distance сенсоры не дают данных")
        
        print()
        
        # 6. Полет по периметру: автоматическое обнаружение остановки перед стеной и поворот направо
        log_info("Начинаем автоматический облет периметра с обнаружением стен...")
        success = fly_perimeter_auto_wall_detection(drone)
        
        if not success:
            log_warn("Облет периметра завершен с предупреждениями, но продолжаем...")
        
        print()
        
        # 8. Возврат в стартовую точку
        if drone.start_position:
            log_info("Возврат в стартовую точку...")
            drone.move_to_safe(drone.start_position[0], drone.start_position[1], FLIGHT_HEIGHT)
            log_ok("Возврат в стартовую точку завершен")
        
        # 9. Посадка
        drone.land()
        
        print()
        
        # 9. Сохраняем результаты и создаем визуализацию маршрута
        json_path, png_path = save_route_results(drone)
        
        print()
        print("=" * 60)
        log_ok("ПОЛЕТ ЗАВЕРШЕН!")
        if json_path:
            print(f"  JSON: {json_path}")
        if png_path:
            print(f"  PNG:  {png_path}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        log_warn("Прервано пользователем")
        drone.hover()
        drone.land()
        # Сохраняем маршрут даже при прерывании
        try:
            json_path, png_path = save_route_results(drone)
            if json_path:
                log_info(f"Маршрут сохранен: {json_path}")
        except Exception:
            pass
    except Exception as e:
        log_error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        try:
            drone.hover()
            drone.land()
            # Пытаемся сохранить маршрут
            json_path, png_path = save_route_results(drone)
            if json_path:
                log_info(f"Маршрут сохранен: {json_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()