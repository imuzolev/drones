"""Open3D visualization utilities."""

from __future__ import annotations

import threading
import time
from typing import Optional, List

from .mapping import PointCloudAccumulator

class RealtimePointCloudVisualizer:
    """
    Класс для визуализации облака точек в реальном времени.
    Окно открывается сразу при создании и обновляется периодически.
    """
    
    def __init__(self, acc: PointCloudAccumulator, update_interval: float = 0.1, max_display_points: int = 100_000):
        """
        Args:
            acc: PointCloudAccumulator для получения точек
            update_interval: интервал обновления визуализации в секундах
            max_display_points: максимальное количество точек для отображения (для производительности)
        """
        self.acc = acc
        self.update_interval = update_interval
        self.max_display_points = max_display_points
        self.vis = None
        self.pcd = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def _transform_ned_to_vis(self, points_xyz):
        """Преобразует точки из NED в систему координат для визуализации."""
        try:
            import numpy as np
        except Exception:
            return None
        
        if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
            return None
        
        # Преобразуем из NED в стандартную систему координат (Z вверх)
        # NED: X=North, Y=East, Z=Down (положительный Z вниз)
        # Для визуализации: X=Right, Y=Forward, Z=Up
        points_vis = np.zeros_like(points_xyz)
        points_vis[:, 0] = points_xyz[:, 1]   # East -> Right
        points_vis[:, 1] = -points_xyz[:, 2]   # -Down -> Up
        points_vis[:, 2] = -points_xyz[:, 0]   # -North -> Forward
        return points_vis
    
    def _update_visualization(self):
        """Обновляет визуализацию с новыми данными из аккумулятора."""
        try:
            import numpy as np
            import open3d as o3d
        except ImportError:
            return False
        
        # Получаем снимок точек из аккумулятора
        points_xyz = self.acc.snapshot(max_points=self.max_display_points)
        
        if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
            # Нет точек, но продолжаем обновлять окно
            with self.lock:
                if self.vis is not None and self.running:
                    try:
                        self.vis.poll_events()
                        self.vis.update_renderer()
                    except:
                        pass
            return False
        
        # Преобразуем координаты
        points_vis = self._transform_ned_to_vis(points_xyz)
        if points_vis is None:
            return False
        
        with self.lock:
            if self.vis is None or not self.running:
                return False
            
            try:
                # Обновляем или создаем облако точек
                if self.pcd is None:
                    self.pcd = o3d.geometry.PointCloud()
                    self.pcd.points = o3d.utility.Vector3dVector(points_vis)
                    # Устанавливаем зеленый цвет
                    colors = np.ones((points_vis.shape[0], 3)) * [0.0, 0.8, 0.0]
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                    self.vis.add_geometry(self.pcd, reset_bounding_box=True)
                    
                    # Настраиваем камеру для просмотра облака точек
                    view_control = self.vis.get_view_control()
                    if points_vis.shape[0] > 0:
                        center = np.mean(points_vis, axis=0)
                        extent = np.max(points_vis, axis=0) - np.min(points_vis, axis=0)
                        max_extent = np.max(extent) if np.max(extent) > 0 else 10.0
                        
                        view_control.set_front([0.5, -0.5, -0.7])
                        view_control.set_lookat(center)
                        view_control.set_up([0, 1, 0])
                        view_control.set_zoom(0.7)
                else:
                    # Обновляем существующее облако точек
                    old_point_count = len(self.pcd.points)
                    self.pcd.points = o3d.utility.Vector3dVector(points_vis)
                    # Обновляем цвета
                    colors = np.ones((points_vis.shape[0], 3)) * [0.0, 0.8, 0.0]
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Если это первое появление точек (было пусто, стало не пусто), настраиваем камеру
                    if old_point_count == 0 and points_vis.shape[0] > 0:
                        # Перестраиваем bounding box и настраиваем камеру
                        self.vis.clear_geometries()
                        self.vis.add_geometry(self.pcd, reset_bounding_box=True)
                        view_control = self.vis.get_view_control()
                        center = np.mean(points_vis, axis=0)
                        extent = np.max(points_vis, axis=0) - np.min(points_vis, axis=0)
                        max_extent = np.max(extent) if np.max(extent) > 0 else 10.0
                        view_control.set_front([0.5, -0.5, -0.7])
                        view_control.set_lookat(center)
                        view_control.set_up([0, 1, 0])
                        view_control.set_zoom(0.7)
                        print(f"[VISUALIZER] Первые точки появились! Центр: {center}, Размер: {extent}")
                    else:
                        # Обычное обновление
                        self.vis.update_geometry(self.pcd)
                
                # Обрабатываем события окна и обновляем рендерер
                self.vis.poll_events()
                self.vis.update_renderer()
                
                return True
            except Exception as e:
                print(f"[VISUALIZER] Ошибка при обновлении геометрии: {e}")
                return False
        
    def _visualization_thread(self):
        """Поток для обновления визуализации."""
        try:
            import numpy as np
            import open3d as o3d
        except ImportError as e:
            print(f"[ERROR] Не удалось импортировать open3d: {e}")
            print("[ERROR] Установите open3d: pip install open3d")
            return
        
        # Создаем визуализатор
        with self.lock:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="Облако точек SLAM - Режим реального времени",
                width=1280,
                height=720,
                visible=True
            )
            
            # Настройка параметров рендеринга
            render_option = self.vis.get_render_option()
            render_option.point_size = 2.0  # Увеличиваем размер точек для лучшей видимости
            render_option.background_color = np.array([0.05, 0.05, 0.05])  # Темный фон
            render_option.show_coordinate_frame = True  # Показать систему координат
            
            # Настраиваем начальную позицию камеры
            view_control = self.vis.get_view_control()
            view_control.set_front([0.5, -0.5, -0.7])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 1, 0])
            view_control.set_zoom(0.5)
            
            # Инициализируем пустое облако точек (будет обновляться когда появятся данные)
            self.pcd = None
        
        print("[VISUALIZER] Окно визуализации открыто. Обновление в реальном времени...")
        print("[VISUALIZER] Управление: мышь - вращение, колесико - масштаб, Shift+мышь - перемещение")
        
        # Основной цикл обновления
        update_count = 0
        while self.running:
            try:
                # Обновляем визуализацию
                updated = self._update_visualization()
                
                # Периодически выводим информацию о количестве точек
                update_count += 1
                if update_count % 50 == 0:  # Каждые 5 секунд (50 * 0.1 сек)
                    try:
                        import numpy as np
                        points_xyz = self.acc.snapshot(max_points=100)
                        if points_xyz is not None and points_xyz.shape[0] > 0:
                            points_vis = self._transform_ned_to_vis(points_xyz)
                            if points_vis is not None:
                                center = np.mean(points_vis, axis=0)
                                extent = np.max(points_vis, axis=0) - np.min(points_vis, axis=0)
                                print(f"[VISUALIZER] Отображается {points_xyz.shape[0]} точек (макс. {self.max_display_points})")
                                print(f"[VISUALIZER] Центр: {center}, Размер: {extent}")
                            else:
                                print(f"[VISUALIZER] Получено {points_xyz.shape[0]} точек, но преобразование не удалось")
                        else:
                            print(f"[VISUALIZER] Ожидание данных от лидара... (накоплено: {self.acc._total_points})")
                    except Exception as e:
                        print(f"[VISUALIZER] Ошибка при выводе информации: {e}")
                
                # Проверяем, закрыто ли окно
                with self.lock:
                    if self.vis is None:
                        break
                    try:
                        if not self.vis.poll_events():
                            # Окно закрыто пользователем
                            self.running = False
                            break
                    except RuntimeError:
                        # Окно было закрыто
                        self.running = False
                        break
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"[VISUALIZER] Ошибка при обновлении: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(self.update_interval)
        
        # Закрываем окно
        with self.lock:
            if self.vis is not None:
                self.vis.destroy_window()
                self.vis = None
        
        print("[VISUALIZER] Визуализация завершена.")
    
    def start(self):
        """Запускает визуализацию в отдельном потоке."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._visualization_thread, daemon=True)
        self.thread.start()
        print("[VISUALIZER] Визуализация запущена в отдельном потоке.")
    
    def stop(self):
        """Останавливает визуализацию."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        print("[VISUALIZER] Визуализация остановлена.")
    
    def is_running(self):
        """Проверяет, работает ли визуализация."""
        return self.running


def display_point_cloud(points_xyz, clean_cloud: bool = True, point_size: float = 1.5, wireframe_paths: Optional[List[str]] = None):
    """
    Отображает облако точек в окне с помощью open3d с улучшенной визуализацией.
    
    Args:
        points_xyz: numpy array (N, 3) точек в системе координат NED
        clean_cloud: если True, применяет очистку облака точек перед визуализацией
        point_size: размер точек для визуализации (по умолчанию 1.5)
        wireframe_paths: список путей к файлам wireframe моделей (PLY, OBJ, STL) для наложения
    """
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as e:
        print(f"[ERROR] Не удалось импортировать open3d: {e}")
        print("[ERROR] Установите open3d: pip install open3d")
        return
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        print("[ERROR] Облако точек пустое!")
        return
    
    print(f"[INFO] Исходное облако точек: {points_xyz.shape[0]} точек")
    
    # Очистка облака точек для улучшения визуализации
    if clean_cloud:
        print("[INFO] Очистка облака точек...")
        points_xyz = _clean_point_cloud(points_xyz, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0)
        if points_xyz is not None and points_xyz.shape[0] > 0:
            print(f"[INFO] После очистки: {points_xyz.shape[0]} точек")
        else:
            print("[WARNING] После очистки облако точек стало пустым, используем исходное")
            points_xyz = points_xyz if points_xyz is not None else points_xyz
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        print("[ERROR] Облако точек пустое после обработки!")
        return
    
    # Преобразуем из NED в стандартную систему координат (Z вверх)
    # NED: X=North, Y=East, Z=Down (положительный Z вниз)
    # Для визуализации: X=Right, Y=Forward, Z=Up
    points_vis = np.zeros_like(points_xyz)
    points_vis[:, 0] = points_xyz[:, 1]   # East -> Right
    points_vis[:, 1] = -points_xyz[:, 2]  # -Down -> Up
    points_vis[:, 2] = -points_xyz[:, 0]  # -North -> Forward
    
    # Создаем облако точек open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_vis)
    
    # Устанавливаем зеленый цвет для всех точек
    if points_vis.shape[0] > 0:
        # Создаем массив зеленого цвета для всех точек
        # RGB: (0.0, 0.8, 0.0) - яркий зеленый цвет
        colors = np.ones((points_vis.shape[0], 3)) * [0.0, 0.8, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Вычисляем нормали для улучшения визуализации (опционально, но улучшает вид)
    print("[INFO] Вычисление нормалей для улучшения визуализации...")
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
    except Exception as e:
        print(f"[WARNING] Не удалось вычислить нормали: {e}")
    
    # Настройка визуализатора с улучшенными параметрами
    print("[INFO] Открывается окно с облаком точек...")
    print("[INFO] Управление: мышь - вращение, колесико - масштаб, Shift+мышь - перемещение")
    print("[INFO] Закройте окно для завершения программы.")
    
    # Загружаем wireframe модели, если они предоставлены
    wireframe_geometries = []
    if wireframe_paths:
        print("[INFO] Загрузка wireframe моделей...")
        wireframe_colors = [
            [1.0, 0.0, 0.0],  # Красный
            [0.0, 0.0, 1.0],  # Синий
            [1.0, 1.0, 1.0],  # Белый
            [0.0, 1.0, 0.0],  # Зеленый
        ]
        for i, wireframe_path in enumerate(wireframe_paths):
            if os.path.exists(wireframe_path):
                try:
                    # Пытаемся загрузить как mesh
                    mesh = o3d.io.read_triangle_mesh(wireframe_path)
                    if len(mesh.vertices) > 0:
                        # Преобразуем в wireframe (только ребра)
                        mesh.compute_vertex_normals()
                        # Устанавливаем цвет
                        color = wireframe_colors[i % len(wireframe_colors)]
                        mesh.paint_uniform_color(color)
                        # Добавляем в список
                        wireframe_geometries.append(mesh)
                        print(f"[INFO] Загружена wireframe модель: {wireframe_path} ({len(mesh.vertices)} вершин)")
                    else:
                        print(f"[WARNING] Пустая wireframe модель: {wireframe_path}")
                except Exception as e:
                    print(f"[WARNING] Не удалось загрузить wireframe модель {wireframe_path}: {e}")
            else:
                print(f"[WARNING] Файл wireframe модели не найден: {wireframe_path}")
    
    # Создаем визуализатор с настройками
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Облако точек SLAM - Детальная визуализация", width=1920, height=1080, visible=True)
    vis.add_geometry(pcd)
    
    # Добавляем wireframe модели, если они есть
    for wireframe in wireframe_geometries:
        vis.add_geometry(wireframe)
    
    # Настройка параметров рендеринга
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Темный фон
    render_option.show_coordinate_frame = True  # Показать систему координат
    if wireframe_geometries:
        # Для wireframe моделей используем wireframe режим
        render_option.mesh_show_wireframe = True
        render_option.mesh_show_back_face = True
    
    # Настройка камеры для лучшего обзора
    view_control = vis.get_view_control()
    
    # Вычисляем центр облака точек
    center = np.mean(points_vis, axis=0)
    # Вычисляем размер облака точек
    extent = np.max(points_vis, axis=0) - np.min(points_vis, axis=0)
    max_extent = np.max(extent)
    
    # Устанавливаем камеру для обзора всего облака
    view_control.set_front([0.5, -0.5, -0.7])
    view_control.set_lookat(center)
    view_control.set_up([0, 1, 0])
    view_control.set_zoom(0.7)
    
    # Запускаем визуализацию
    vis.run()
    vis.destroy_window()
    
    print("[INFO] Визуализация завершена.")


