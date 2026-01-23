# Инструкции по установке зависимостей

## ⚠️ Важное замечание о версии Python

**Проблема**: В системе установлен только Python 3.13, но библиотека `open3d` (требуется для `lidar_3d_mapping.py` и `view_ply.py`) поддерживает только Python 3.8-3.12.

## Что было сделано

1. ✅ Создано виртуальное окружение `venv` (с Python 3.13)
2. ✅ Установлен `numpy` версии 2.4.1
3. ❌ Не удалось установить `open3d` (несовместимость с Python 3.13)
4. ❌ Не удалось установить `projectairsim` (зависит от `open3d`)

## Решение: установить Python 3.12

### Вариант 1: Использовать Python 3.12 (рекомендуется)

1. Установите Python 3.12 с [python.org](https://www.python.org/downloads/) или через winget:
   ```powershell
   winget install Python.Python.3.12
   ```

2. Создайте новое виртуальное окружение с Python 3.12:
   ```powershell
   py -3.12 -m venv venv
   ```

3. Активируйте виртуальное окружение:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. Установите зависимости:
   ```powershell
   pip install -r requirements.txt
   pip install -e ./client/python/projectairsim
   ```

### Вариант 2: Использовать готовый скрипт (после установки Python 3.12)

Просто запустите:
```powershell
.\setup_venv.bat
```

## Необходимые зависимости

Основные зависимости (указаны в `requirements.txt`):
- `numpy>=1.21.0` - для работы с массивами
- `open3d>=0.16.0,<0.17.0` - для работы с облаками точек
- `projectairsim` - клиент для ProjectAirSim (устанавливается из `./client/python/projectairsim`)

ProjectAirSim также автоматически установит:
- `pynng`, `msgpack`, `opencv-python`, `matplotlib`, и другие зависимости

## Текущий статус

Виртуальное окружение создано, но **не полностью настроено** из-за несовместимости версии Python с `open3d`.

Для полной установки необходимо использовать Python 3.12 или более раннюю версию.
