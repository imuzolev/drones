@echo off
REM Скрипт для создания виртуального окружения с Python 3.12
REM Open3D не поддерживает Python 3.13, требуется Python 3.8-3.12

echo [INFO] Проверка версии Python...
python --version

echo [INFO] Создание виртуального окружения...
python -m venv venv

echo [INFO] Активация виртуального окружения...
call venv\Scripts\activate.bat

echo [INFO] Обновление pip...
python -m pip install --upgrade pip

echo [INFO] Установка зависимостей из requirements.txt...
pip install -r requirements.txt

echo [INFO] Установка ProjectAirSim в режиме разработки...
pip install -e ./client/python/projectairsim

echo [INFO] Установка завершена!
echo [INFO] Для активации виртуального окружения выполните: venv\Scripts\activate.bat
