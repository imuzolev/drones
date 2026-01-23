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
    
    def __init__(self, acc: PointCloudAccumulator, pose_provider=None, update_interval: float = 0.1, max_display_points: int = 500_000):
        """
        Args:
            acc: PointCloudAccumulator для получения точек
            pose_provider: объект с методом snapshot() -> (pose_msg, timestamp), например PoseHistory или PoseLatest
            update_interval: интервал обновления визуализации в секундах
            max_display_points: максимальное количество точек для отображения
        """
        self.acc = acc
        self.pose_provider = pose_provider
        self.update_interval = update_interval
        self.max_display_points = max_display_points
        self.vis = None
        self.pcd = None
        self.drone_frame = None  # Coordinate frame for the drone
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

    def _get_drone_transform_vis(self):
        """Возвращает матрицу трансформации (4x4) дрона в системе координат визуализации."""
        if self.pose_provider is None:
            return None
            
        try:
            import numpy as np
            pose_msg, _ = self.pose_provider.snapshot()
            if not pose_msg or not isinstance(pose_msg, dict):
                return None
                
            pos = pose_msg.get("position", {})
            ori = pose_msg.get("orientation", {})
            
            # Position NED
            pn = float(pos.get("x", 0.0))
            pe = float(pos.get("y", 0.0))
            pd = float(pos.get("z", 0.0))
            
            # Orientation NED (Quaternion)
            w = float(ori.get("w", 1.0))
            x = float(ori.get("x", 0.0))
            y = float(ori.get("y", 0.0))
            z = float(ori.get("z", 0.0))
            
            # Rotation Matrix Body -> NED
            # R = [1-2(y²+z²)   2(xy-wz)     2(xz+wy)   ]
            #     [2(xy+wz)     1-2(x²+z²)   2(yz-wx)   ]
            #     [2(xz-wy)     2(yz+wx)     1-2(x²+y²) ]
            
            xx = x*x; yy = y*y; zz = z*z
            xy = x*y; xz = x*z; yz = y*z
            wx = w*x; wy = w*y; wz = w*z
            
            R_ned = np.array([
                [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
                [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
                [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
            ])
            
            # Transform to Vis Frame
            # Basis vectors of Body in NED are the columns of R_ned
            # b_x_ned = R_ned[:, 0]
            # b_y_ned = R_ned[:, 1]
            # b_z_ned = R_ned[:, 2]
            
            # NED to Vis mapping for vectors: v_vis = [v_ned.y, -v_ned.z, -v_ned.x]
            def to_vis_vec(v):
                return np.array([v[1], -v[2], -v[0]])
                
            b_x_vis = to_vis_vec(R_ned[:, 0])
            b_y_vis = to_vis_vec(R_ned[:, 1])
            b_z_vis = to_vis_vec(R_ned[:, 2])
            
            # Rotation Matrix Vis
            R_vis = np.column_stack([b_x_vis, b_y_vis, b_z_vis])
            
            # Position Vis
            p_vis = np.array([pe, -pd, -pn])
            
            # 4x4 Transform
            T = np.eye(4)
            T[:3, :3] = R_vis
            T[:3, 3] = p_vis
            
            return T
            
        except Exception:
            return None
    
    def _update_visualization(self):
        """Обновляет визуализацию с новыми данными из аккумулятора."""
        try:
            import numpy as np
            import open3d as o3d
        except ImportError:
            return False
        
        # Получаем снимок точек из аккумулятора
        points_xyz = self.acc.snapshot(max_points=self.max_display_points)
        
        # Обновляем дрон
        drone_T = self._get_drone_transform_vis()
        
        with self.lock:
            if self.vis is None or not self.running:
                return False
                
            try:
                # 1. Update Point Cloud
                has_points = (points_xyz is not None and getattr(points_xyz, "size", 0) > 0)
                
                if has_points:
                    points_vis = self._transform_ned_to_vis(points_xyz)
                    
                    # === Color classification: Floor (gray), Walls (blue), Obstacles (green) ===
                    vis_x = points_vis[:, 0]  # East/Right
                    vis_y = points_vis[:, 1]  # Up (height)
                    vis_z = points_vis[:, 2]  # North/Forward
                    
                    # 1. Floor: lowest points (within 0.5m from minimum height)
                    min_y = np.min(vis_y)
                    floor_mask = (vis_y < min_y + 0.5)
                    
                    # 2. Walls: points near the boundaries of the map (outer perimeter)
                    min_x, max_x = np.min(vis_x), np.max(vis_x)
                    min_z, max_z = np.min(vis_z), np.max(vis_z)
                    
                    wall_margin = 1.5
                    near_min_x = (vis_x < min_x + wall_margin)
                    near_max_x = (vis_x > max_x - wall_margin)
                    near_min_z = (vis_z < min_z + wall_margin)
                    near_max_z = (vis_z > max_z - wall_margin)
                    
                    wall_mask = (near_min_x | near_max_x | near_min_z | near_max_z) & ~floor_mask
                    
                    # 3. Obstacles: everything else
                    obstacle_mask = ~floor_mask & ~wall_mask
                    
                    colors = np.zeros((points_vis.shape[0], 3))
                    colors[floor_mask] = [0.4, 0.4, 0.4]      # Floor (Gray)
                    colors[wall_mask] = [0.2, 0.4, 1.0]       # Walls (Blue)
                    colors[obstacle_mask] = [0.0, 1.0, 0.0]   # Obstacles (Green)

                    if self.pcd is None:
                        self.pcd = o3d.geometry.PointCloud()
                        self.pcd.points = o3d.utility.Vector3dVector(points_vis)
                        self.pcd.colors = o3d.utility.Vector3dVector(colors)
                        self.vis.add_geometry(self.pcd, reset_bounding_box=False)
                    else:
                        self.pcd.points = o3d.utility.Vector3dVector(points_vis)
                        self.pcd.colors = o3d.utility.Vector3dVector(colors)
                        self.vis.update_geometry(self.pcd)

                # 2. Update Drone Marker
                if drone_T is not None:
                    if self.drone_frame is None:
                        # Create drone coordinate frame (Red=Forward, Green=Right, Blue=Down in body frame if mapped correctly)
                        # Actually CreateCoordinateFrame makes X=Red, Y=Green, Z=Blue
                        # Our transform maps BodyX->Red, BodyY->Green, BodyZ->Blue
                        self.drone_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                        self.drone_frame.transform(drone_T)
                        self.vis.add_geometry(self.drone_frame, reset_bounding_box=False)
                    else:
                        # Since open3d geometry transform is cumulative or applied to vertices,
                        # it's easier to recreate or reset. 
                        # But create_coordinate_frame is a Mesh. We can set vertices?
                        # No, easier to remove and add, or use a persistent transform if possible.
                        # Actually Open3D legacy visualizer is tricky with transforms.
                        # Easiest: Create new frame, or deep copy canonical and transform.
                        # But removing/adding is slow.
                        # Let's try transforming vertices?
                        # No, let's keep a canonical frame and transform it.
                        
                        # Better approach: store canonical frame and copy+transform?
                        # No, visualizer needs the object reference.
                        
                        # We can modify the vertices of self.drone_frame in place?
                        # Let's try removing and adding for simplicity first, performance might be ok for 1 object.
                        self.vis.remove_geometry(self.drone_frame, reset_bounding_box=False)
                        self.drone_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                        self.drone_frame.transform(drone_T)
                        self.vis.add_geometry(self.drone_frame, reset_bounding_box=False)

                # 3. First time camera setup
                # Only if we have points and camera hasn't been set
                if has_points and not hasattr(self, "_camera_initialized"):
                    self.vis.reset_view_point(True)
                    view_control = self.vis.get_view_control()
                    center = np.mean(points_vis, axis=0)
                    view_control.set_front([0.5, -0.5, -0.7])
                    view_control.set_lookat(center)
                    view_control.set_up([0, 1, 0])
                    view_control.set_zoom(0.7)
                    self._camera_initialized = True

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
    
    # === Color classification: Floor (gray), Walls (blue), Obstacles (green) ===
    if points_vis.shape[0] > 0:
        vis_x = points_vis[:, 0]  # East/Right
        vis_y = points_vis[:, 1]  # Up (height)
        vis_z = points_vis[:, 2]  # North/Forward
        
        # 1. Floor: lowest points (within 0.5m from minimum height)
        min_y = np.min(vis_y)
        floor_mask = (vis_y < min_y + 0.5)
        
        # 2. Walls: points near the boundaries of the map (outer perimeter)
        min_x, max_x = np.min(vis_x), np.max(vis_x)
        min_z, max_z = np.min(vis_z), np.max(vis_z)
        
        wall_margin = 1.5
        near_min_x = (vis_x < min_x + wall_margin)
        near_max_x = (vis_x > max_x - wall_margin)
        near_min_z = (vis_z < min_z + wall_margin)
        near_max_z = (vis_z > max_z - wall_margin)
        
        wall_mask = (near_min_x | near_max_x | near_min_z | near_max_z) & ~floor_mask
        
        # 3. Obstacles: everything else
        obstacle_mask = ~floor_mask & ~wall_mask
        
        colors = np.zeros((points_vis.shape[0], 3))
        colors[floor_mask] = [0.4, 0.4, 0.4]      # Floor (Gray)
        colors[wall_mask] = [0.2, 0.4, 1.0]       # Walls (Blue)
        colors[obstacle_mask] = [0.0, 1.0, 0.0]   # Obstacles (Green)
        
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


