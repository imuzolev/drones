"""
Утилита для исправления прав доступа к settings.json
Запустите этот скрипт, если Unreal Engine выдаёт ошибку "Permission denied"
"""
import os
import sys
import stat
import subprocess
import codecs

# Настройка кодировки для Windows терминала
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def fix_settings_permissions():
    """Снимает read-only атрибут с settings.json"""
    settings_path = os.path.expanduser(r"~\Documents\AirSim\settings.json")
    
    if not os.path.exists(settings_path):
        print(f"[ERROR] Файл не найден: {settings_path}")
        return False
    
    print(f"Исправление прав доступа к файлу: {settings_path}")
    
    # Способ 1: через os.chmod
    try:
        current_mode = os.stat(settings_path).st_mode
        os.chmod(settings_path, current_mode | stat.S_IWRITE | stat.S_IREAD)
        print("[OK] Права доступа исправлены через os.chmod")
    except Exception as e:
        print(f"[WARN] Не удалось через os.chmod: {e}")
    
    # Способ 2: через Windows attrib (если первый способ не сработал)
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['attrib', '-R', settings_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("[OK] Права доступа исправлены через attrib")
            else:
                print(f"[WARN] attrib вернул код: {result.returncode}")
        except Exception as e:
            print(f"[WARN] Не удалось через attrib: {e}")
    
    # Проверяем результат
    try:
        file_stat = os.stat(settings_path)
        is_readable = bool(file_stat.st_mode & stat.S_IREAD)
        is_writable = bool(file_stat.st_mode & stat.S_IWRITE)
        
        if is_readable and is_writable:
            print("[OK] Файл доступен для чтения и записи")
            return True
        else:
            print(f"[WARN] Файл: читаемый={is_readable}, записываемый={is_writable}")
            return False
    except Exception as e:
        print(f"[ERROR] Ошибка проверки прав: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Исправление прав доступа к settings.json")
    print("=" * 60)
    print()
    
    if fix_settings_permissions():
        print()
        print("[OK] Готово! Теперь можно запускать Unreal Engine.")
    else:
        print()
        print("[WARN] Возможны проблемы с правами доступа.")
        print("       Попробуйте запустить скрипт от имени администратора.")

