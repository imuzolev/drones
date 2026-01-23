@echo off
:: Скрипт для запуска 3D картографирования с помощью Lidar
:: Copyright (C) 2025

echo ============================================================
echo   LIDAR 3D MAPPING - ProjectAirSim Blocks
echo ============================================================
echo.

:: Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден!
    echo Пожалуйста, установите Python 3.8+ и добавьте его в PATH.
    pause
    exit /b 1
)

:: Переходим в директорию проекта
cd /d "%~dp0"

:: Активируем виртуальное окружение если есть
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Активация виртуального окружения...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Активация виртуального окружения...
    call .venv\Scripts\activate.bat
)

:: Устанавливаем необходимые пакеты
echo [INFO] Проверка и установка необходимых пакетов...
pip install open3d matplotlib numpy pynng msgpack cryptography >nul 2>&1

:: Устанавливаем projectairsim если не установлен
pip show projectairsim >nul 2>&1
if errorlevel 1 (
    echo [INFO] Установка projectairsim...
    pip install -e client\python\projectairsim >nul 2>&1
)

echo.
echo [INFO] Запуск программы 3D картографирования...
echo.
echo ВАЖНО: Перед запуском убедитесь, что:
echo   1. Unreal Engine с проектом Blocks запущен
echo   2. В UE4 нажата кнопка Play
echo.
echo Если симулятор ещё не запущен, программа попытается
echo найти и запустить его автоматически.
echo.

:: Запускаем основной скрипт
python lidar_3d_mapping.py

echo.
echo [INFO] Программа завершена.
pause
