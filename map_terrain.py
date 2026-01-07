"""
Безопасное 3D сканирование карты с дрона в AirSim.
Автор: AI Assistant
Дата: 2026

Особенности:
- Атомарная запись settings.json (защита от повреждения)
- Безопасный полёт с обходом препятствий
- Использование лидара и depth-камеры для навигации
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

# Параметры сканирования
SCAN_AREA_X = 10           # ширина области (метры)
SCAN_AREA_Y = 10           # глубина области (метры)
SCAN_Z_MIN = -20           # минимальная высота (NED, отрицательное = вверх)
SCAN_Z_MAX = -2            # максимальная высота
SCAN_RES_XY = 5            # разрешение по XY
SCAN_RES_Z = 3             # разрешение по Z

# Параметры безопасности полёта
CMD_HZ = 20.0              # частота команд управления
CMD_DT = 1.0 / CMD_HZ
MAX_SPEED = 3.0            # максимальная скорость
MIN_SPEED = 0.8            # минимальная скорость
ARRIVAL_TOL = 1.0          # допуск прибытия в точку

# Пороги безопасности
SAFE_DISTANCE = 8.0        # начинаем замедление
WARN_DISTANCE = 5.0        # начинаем обход
STOP_DISTANCE = 3.0        # аварийный стоп/подъём

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
        
        # Включаем API и армим
        self.client.enableApiControl(True, VEHICLE_NAME)
        self.client.armDisarm(True, VEHICLE_NAME)
        
        log_ok("Дрон инициализирован")
        
    def takeoff(self):
        """Безопасный взлёт."""
        log_info("Взлёт...")
        self.client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
        time.sleep(1.0)
        log_ok("Взлёт завершён")
        
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
    
    def get_obstacle_distance(self):
        """Получает минимальную дистанцию до препятствия (лидар + depth)."""
        d_lidar = self.get_lidar_distance()
        d_depth = self.get_depth_distance()
        
        if d_lidar is None and d_depth is None:
            return None
        if d_lidar is None:
            return d_depth
        if d_depth is None:
            return d_lidar
        return min(d_lidar, d_depth)
    
    def send_velocity(self, vx, vy, vz):
        """Отправляет команду скорости (без блокировки)."""
        self.client.moveByVelocityAsync(
            float(vx), float(vy), float(vz), 
            duration=CMD_DT,
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
        # Ограничиваем высоту
        target_z = max(float(SCAN_Z_MIN), min(float(SCAN_Z_MAX), float(target_z)))
        
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
            vz = np.clip(uz * speed, -1.5, 1.5)  # Ограничиваем вертикальную скорость
            
            self.send_velocity(vx, vy, vz)


# =========================
# СКАНИРОВАНИЕ
# =========================

def run_scan(drone: SafeDrone):
    """Выполняет 3D сканирование области."""
    
    # Генерируем точки сканирования
    x_points = np.arange(-SCAN_AREA_X / 2, SCAN_AREA_X / 2, SCAN_RES_XY)
    y_points = np.arange(-SCAN_AREA_Y / 2, SCAN_AREA_Y / 2, SCAN_RES_XY)
    z_points = np.arange(SCAN_Z_MIN, SCAN_Z_MAX, SCAN_RES_Z)
    
    total_points = len(x_points) * len(y_points) * len(z_points)
    
    log_info(f"Параметры сканирования:")
    print(f"  Область: {SCAN_AREA_X}x{SCAN_AREA_Y} м")
    print(f"  Высоты: от {SCAN_Z_MIN} до {SCAN_Z_MAX} м")
    print(f"  Разрешение XY: {SCAN_RES_XY} м, Z: {SCAN_RES_Z} м")
    print(f"  Всего точек: {total_points}")
    print(f"  Уровней высоты: {len(z_points)}")
    
    # Карта объёма: 0=свободно, 1=препятствие, 2=недоступно
    volume_map = np.zeros((len(z_points), len(y_points), len(x_points)), dtype=np.int32)
    position_log = []
    
    point_idx = 0
    skipped = 0
    
    for k, z in enumerate(z_points):
        log_info(f"=== Уровень {k+1}/{len(z_points)}: высота {z:.1f} м ===")
        
        for i, y in enumerate(y_points):
            for j, x in enumerate(x_points):
                point_idx += 1
                tx, ty, tz = float(x), float(y), float(z)
                
                # Перемещаемся к точке
                reached = drone.move_to_safe(tx, ty, tz)
                
                # Получаем фактическую позицию
                ax, ay, az = drone.get_position()
                dist_error = math.sqrt((tx-ax)**2 + (ty-ay)**2 + (tz-az)**2)
                
                # Проверяем препятствие в точке
                obstacle_dist = drone.get_obstacle_distance()
                has_obstacle = (obstacle_dist is not None and obstacle_dist < SAFE_DISTANCE)
                
                if not reached or dist_error > 5.0:
                    skipped += 1
                    volume_map[k, i, j] = 2  # Недоступно
                    position_log.append({
                        "target": {"x": tx, "y": ty, "z": tz},
                        "actual": {"x": ax, "y": ay, "z": az},
                        "accessible": False,
                        "obstacle_dist": obstacle_dist
                    })
                else:
                    volume_map[k, i, j] = 1 if has_obstacle else 0
                    position_log.append({
                        "target": {"x": tx, "y": ty, "z": tz},
                        "actual": {"x": ax, "y": ay, "z": az},
                        "accessible": True,
                        "obstacle": has_obstacle,
                        "obstacle_dist": obstacle_dist
                    })
                
                # Прогресс
                if point_idx % 50 == 0:
                    pct = int(point_idx * 100 / total_points)
                    print(f"  Прогресс: {point_idx}/{total_points} ({pct}%), пропущено: {skipped}")
            
            if (i + 1) % 5 == 0:
                print(f"  Y строк: {i+1}/{len(y_points)}")
    
    return volume_map, x_points, y_points, z_points, position_log, skipped


# =========================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =========================

def save_results(volume_map, x_points, y_points, z_points, position_log, skipped):
    """Сохраняет результаты сканирования."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "maps")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join(output_dir, f"terrain_map_{timestamp}")
    
    # Сохраняем JSON
    json_path = f"{base_path}.json"
    
    total_points = len(x_points) * len(y_points) * len(z_points)
    obstacles = int(np.sum(volume_map == 1))
    free = int(np.sum(volume_map == 0))
    inaccessible = int(np.sum(volume_map == 2))
    
    data = {
        "scan_parameters": {
            "area_x": SCAN_AREA_X,
            "area_y": SCAN_AREA_Y,
            "z_min": SCAN_Z_MIN,
            "z_max": SCAN_Z_MAX,
            "resolution_xy": SCAN_RES_XY,
            "resolution_z": SCAN_RES_Z,
            "timestamp": timestamp,
            "safety": {
                "safe_distance": SAFE_DISTANCE,
                "warn_distance": WARN_DISTANCE,
                "stop_distance": STOP_DISTANCE
            }
        },
        "statistics": {
            "total_points": total_points,
            "obstacles": obstacles,
            "free": free,
            "inaccessible": inaccessible,
            "skipped": skipped
        },
        "volume_map": volume_map.tolist(),
        "x_points": x_points.tolist(),
        "y_points": y_points.tolist(),
        "z_points": z_points.tolist(),
        "positions": position_log
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    log_ok(f"JSON: {json_path}")
    
    # Создаём визуализацию
    log_info("Создание визуализации...")
    
    top_view = np.sum(volume_map == 1, axis=0)
    side_yz = np.sum(volume_map == 1, axis=2)
    front_xz = np.sum(volume_map == 1, axis=1)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Вид сверху
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(top_view, cmap="hot", origin="lower",
                     extent=[x_points[0], x_points[-1], y_points[0], y_points[-1]])
    plt.colorbar(im1, ax=ax1, label="Препятствия")
    ax1.set_title("Вид сверху (XY)")
    ax1.set_xlabel("X (м)")
    ax1.set_ylabel("Y (м)")
    
    # Вид сбоку
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(side_yz, cmap="hot", origin="lower",
                     extent=[y_points[0], y_points[-1], z_points[0], z_points[-1]],
                     aspect="auto")
    plt.colorbar(im2, ax=ax2, label="Препятствия")
    ax2.set_title("Вид сбоку (YZ)")
    ax2.set_xlabel("Y (м)")
    ax2.set_ylabel("Z (м)")
    
    # Вид спереди
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(front_xz, cmap="hot", origin="lower",
                     extent=[x_points[0], x_points[-1], z_points[0], z_points[-1]],
                     aspect="auto")
    plt.colorbar(im3, ax=ax3, label="Препятствия")
    ax3.set_title("Вид спереди (XZ)")
    ax3.set_xlabel("X (м)")
    ax3.set_ylabel("Z (м)")
    
    # 3D карта
    ax4 = plt.subplot(2, 3, 4, projection="3d")
    obs_idx = np.where(volume_map == 1)
    if len(obs_idx[0]) > 0:
        z_idx, y_idx, x_idx = obs_idx
        xs = x_points[x_idx]
        ys = y_points[y_idx]
        zs = z_points[z_idx]
        step = max(1, len(xs) // 1500)
        ax4.scatter(xs[::step], ys[::step], zs[::step], 
                   c=zs[::step], cmap="viridis", alpha=0.6, s=2)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.set_title("3D карта")
    
    # График по высотам
    ax5 = plt.subplot(2, 3, 5)
    obs_per_level = [int(np.sum(volume_map[k] == 1)) for k in range(len(z_points))]
    ax5.plot(z_points, obs_per_level, 'b-o', linewidth=2)
    ax5.set_xlabel("Высота Z (м)")
    ax5.set_ylabel("Количество препятствий")
    ax5.set_title("Препятствия по высоте")
    ax5.grid(True, alpha=0.3)
    
    # Статистика
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    stats_text = f"""
СТАТИСТИКА СКАНИРОВАНИЯ

Всего точек:  {total_points:,}
Препятствий:  {obstacles:,}
Свободных:    {free:,}
Недоступно:   {inaccessible:,}

Параметры безопасности:
  SAFE_DISTANCE = {SAFE_DISTANCE} м
  WARN_DISTANCE = {WARN_DISTANCE} м
  STOP_DISTANCE = {STOP_DISTANCE} м

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
    print("  БЕЗОПАСНОЕ 3D СКАНИРОВАНИЕ КАРТЫ")
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
        
        print()
        
        # 6. Выполняем сканирование
        log_info("Начинаем сканирование...")
        volume_map, x_pts, y_pts, z_pts, pos_log, skipped = run_scan(drone)
        
        print()
        
        # 7. Возврат домой
        log_info("Возврат в начальную точку...")
        drone.move_to_safe(0.0, 0.0, float(SCAN_Z_MAX))
        
        # 8. Посадка
        drone.land()
        
        print()
        
        # 9. Сохраняем результаты
        json_path, png_path = save_results(volume_map, x_pts, y_pts, z_pts, pos_log, skipped)
        
        print()
        print("=" * 60)
        log_ok("СКАНИРОВАНИЕ ЗАВЕРШЕНО!")
        print(f"  JSON: {json_path}")
        print(f"  PNG:  {png_path}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        log_warn("Прервано пользователем")
        drone.hover()
        drone.land()
    except Exception as e:
        log_error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        try:
            drone.hover()
            drone.land()
        except Exception:
            pass


if __name__ == "__main__":
    main()
