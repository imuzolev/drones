"""Path tracking utilities."""

from __future__ import annotations

from typing import Optional, Tuple, List

from .bootstrap import World

class PathTracker:
    """
    Класс для отслеживания пути дрона и отрисовки красной линии траектории.
    """
    def __init__(self, world: World, min_distance: float = 0.1):
        """
        Инициализация трекера пути.
        
        Args:
            world: Объект World для отрисовки линий
            min_distance: Минимальное расстояние между точками для добавления (м)
        """
        self.world = world
        self.min_distance = min_distance
        self.path_points: List[List[float]] = []
        self.last_point: Optional[List[float]] = None
        
    def update_position(self, position_ned: Tuple[float, float, float]) -> None:
        """
        Обновляет путь дрона и рисует красную линию траектории.
        
        Args:
            position_ned: Позиция дрона в системе NED (north, east, down)
        """
        n, e, d = position_ned
        new_point = [float(n), float(e), float(d)]
        
        # Проверяем расстояние до последней точки
        if self.last_point is not None:
            import math
            dist = math.sqrt(
                (new_point[0] - self.last_point[0]) ** 2 +
                (new_point[1] - self.last_point[1]) ** 2 +
                (new_point[2] - self.last_point[2]) ** 2
            )
            if dist < self.min_distance:
                return  # Пропускаем точку, если она слишком близко
        
        # Добавляем точку
        self.path_points.append(new_point)
        self.last_point = new_point
        
        # Обновляем красную линию на экране
        if len(self.path_points) >= 2:
            try:
                # Красный цвет: [R, G, B, A] = [1.0, 0.0, 0.0, 1.0]
                red_color = [1.0, 0.0, 0.0, 1.0]
                thickness = 3.0  # Толщина линии
                duration = 0.0  # 0 означает, что используется is_persistent
                is_persistent = True  # Линия остается на экране до явного удаления
                
                # Рисуем всю линию заново (это эффективно для persistent маркеров)
                self.world.plot_debug_solid_line(
                    points=self.path_points,
                    color_rgba=red_color,
                    thickness=thickness,
                    duration=duration,
                    is_persistent=is_persistent
                )
            except Exception as e:
                # Игнорируем ошибки отрисовки, чтобы не прерывать полет
                pass
    
    def clear_path(self) -> None:
        """Очищает путь дрона."""
        self.path_points.clear()
        self.last_point = None
        try:
            self.world.flush_persistent_markers()
        except Exception:
            pass


# ============================================================================
# ФУНКЦИИ ОБРАБОТКИ ОБЛАКА ТОЧЕК (интегрировано из Autonomous-Drone-Scanning-and-Mapping)
# ============================================================================

