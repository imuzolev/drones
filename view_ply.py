"""
Простой скрипт для просмотра PLY файла с облаком точек.
Использование: python view_ply.py [путь_к_файлу.ply]
"""

import sys
import argparse

def view_ply_file(filepath: str):
    """
    Открывает и отображает PLY файл с облаком точек.
    
    Args:
        filepath: путь к PLY файлу
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[ERROR] Не удалось импортировать open3d")
        print("[ERROR] Установите open3d: pip install open3d")
        return False
    
    try:
        print(f"[INFO] Загрузка файла: {filepath}")
        pcd = o3d.io.read_point_cloud(filepath)
        
        if len(pcd.points) == 0:
            print("[ERROR] Файл пустой или не содержит точек!")
            return False
        
        print(f"[INFO] Загружено {len(pcd.points)} точек")
        
        # Проверяем наличие цветов
        if len(pcd.colors) > 0:
            print("[INFO] Файл содержит цвета точек")
        else:
            print("[INFO] Файл не содержит цветов, будут использованы цвета по умолчанию")
            # Устанавливаем зеленый цвет по умолчанию
            import numpy as np
            colors = np.ones((len(pcd.points), 3)) * [0.0, 0.8, 0.0]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Вычисляем нормали для лучшей визуализации (опционально)
        try:
            print("[INFO] Вычисление нормалей...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.normalize_normals()
        except Exception as e:
            print(f"[WARNING] Не удалось вычислить нормали: {e}")
        
        # Настройка визуализатора
        print("[INFO] Открывается окно просмотра...")
        print("[INFO] Управление:")
        print("  - Мышь: вращение камеры")
        print("  - Колесико мыши: масштабирование")
        print("  - Shift + мышь: перемещение камеры")
        print("  - Закройте окно для выхода")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Просмотр: {filepath}",
            width=1280,
            height=720,
            visible=True
        )
        vis.add_geometry(pcd, reset_bounding_box=True)
        
        # Настройка параметров рендеринга
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = [0.05, 0.05, 0.05]  # Темный фон
        render_option.show_coordinate_frame = True
        
        # Настройка камеры
        view_control = vis.get_view_control()
        # Вычисляем центр облака точек
        import numpy as np
        points_array = np.asarray(pcd.points)
        center = np.mean(points_array, axis=0)
        extent = np.max(points_array, axis=0) - np.min(points_array, axis=0)
        
        view_control.set_front([0.5, -0.5, -0.7])
        view_control.set_lookat(center)
        view_control.set_up([0, 1, 0])
        view_control.set_zoom(0.7)
        
        # Запускаем визуализацию
        vis.run()
        vis.destroy_window()
        
        print("[INFO] Просмотр завершен.")
        return True
        
    except FileNotFoundError:
        print(f"[ERROR] Файл не найден: {filepath}")
        return False
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке/просмотре файла: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Просмотр PLY файла с облаком точек"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default="result.ply",
        help="Путь к PLY файлу (по умолчанию: result.ply)"
    )
    
    args = parser.parse_args()
    
    if not view_ply_file(args.filepath):
        sys.exit(1)


if __name__ == "__main__":
    main()
