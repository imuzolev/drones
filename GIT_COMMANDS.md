# Основные команды Git

Краткий справочник по работе с Git и GitHub.

## 📋 Базовые команды

### Инициализация репозитория
```bash
git init                          # Инициализация нового репозитория
git clone <url>                   # Клонирование существующего репозитория
```

### Проверка статуса
```bash
git status                        # Показать статус изменений
git log                           # История коммитов
git log --oneline                 # Краткая история коммитов
git diff                          # Показать изменения
git diff <файл>                   # Изменения в конкретном файле
```

## 📝 Работа с файлами

### Добавление файлов
```bash
git add .                         # Добавить все файлы
git add <файл>                    # Добавить конкретный файл
git add *.py                      # Добавить все .py файлы
git add -A                        # Добавить все изменения (включая удалённые)
```

### Коммиты
```bash
git commit -m "Сообщение"         # Создать коммит
git commit -am "Сообщение"        # Добавить и закоммитить (только изменённые файлы)
git commit --amend                # Изменить последний коммит
git commit --amend --no-edit      # Изменить последний коммит без изменения сообщения
```

## 🔄 Работа с удалённым репозиторием

### Подключение к GitHub
```bash
git remote add origin <url>       # Добавить удалённый репозиторий
git remote -v                     # Показать все удалённые репозитории
git remote remove origin          # Удалить удалённый репозиторий
```

### Отправка изменений
```bash
git push                          # Отправить изменения (если upstream настроен)
git push -u origin main           # Отправить и настроить upstream
git push origin <ветка>           # Отправить конкретную ветку
git push --force                  # Принудительная отправка (ОСТОРОЖНО!)
```

### Получение изменений
```bash
git pull                          # Получить и объединить изменения
git fetch                         # Получить изменения без объединения
git fetch origin                  # Получить изменения из origin
```

## 🌿 Работа с ветками

### Создание и переключение
```bash
git branch                        # Показать все ветки
git branch <имя>                  # Создать новую ветку
git checkout <ветка>              # Переключиться на ветку
git checkout -b <имя>             # Создать и переключиться на ветку
git switch <ветка>                 # Переключиться на ветку (новый способ)
git switch -c <имя>               # Создать и переключиться (новый способ)
```

### Переименование и удаление
```bash
git branch -m <новое_имя>         # Переименовать текущую ветку
git branch -d <ветка>             # Удалить ветку (безопасно)
git branch -D <ветка>             # Удалить ветку (принудительно)
```

## ⚙️ Настройка Git

### Глобальная настройка
```bash
git config --global user.name "Имя"
git config --global user.email "email@example.com"
git config --global --list       # Показать все глобальные настройки
```

### Локальная настройка (только для репозитория)
```bash
git config user.name "Имя"
git config user.email "email@example.com"
```

### Создание алиасов
```bash
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
```

## 🔧 Полезные команды

### Отмена изменений
```bash
git restore <файл>                # Отменить изменения в файле
git restore --staged <файл>       # Убрать файл из staging
git reset HEAD <файл>             # Убрать файл из staging (старый способ)
git reset --hard                  # ОТМЕНИТЬ ВСЕ ИЗМЕНЕНИЯ (ОСТОРОЖНО!)
```

### Удаление файлов
```bash
git rm <файл>                     # Удалить файл из Git и файловой системы
git rm --cached <файл>            # Удалить из Git, но оставить локально
git rm -r <папка>                 # Удалить папку
```

### Просмотр истории
```bash
git log --graph --oneline --all   # Красивая визуализация истории
git log --author="Имя"             # Коммиты конкретного автора
git log --since="2 weeks ago"     # Коммиты за период
git show <commit_hash>             # Показать детали коммита
```

## 🚀 Быстрый workflow

### Обновление одного файла
```bash
git add map_terrain.py
git commit -m "Update map_terrain.py"
git push
```

### Обновление всех изменений
```bash
git add .
git commit -m "Описание изменений"
git push
```

### Получение последних изменений
```bash
git pull
```

## 📌 Полезные алиасы (уже настроены)

```bash
git quick "сообщение"              # Добавить все файлы и закоммитить
git pushf                          # Безопасный force push
```

## ⚠️ Важные замечания

- **НИКОГДА** не делайте `git push --force` в основной ветке без необходимости
- Всегда проверяйте `git status` перед коммитом
- Используйте понятные сообщения коммитов
- Регулярно делайте `git pull` перед началом работы
- Не коммитьте файлы с паролями и секретами

## 🔗 Полезные ссылки

- [Официальная документация Git](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

