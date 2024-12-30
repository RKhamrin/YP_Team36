# Используем конкретную версию базового образа для повторяемости сборки
FROM python:3.10

# Указываем рабочую директорию
WORKDIR /app

# Копируем только requirements.txt на раннем этапе
COPY requirements.txt .

# Устанавливаем зависимости, обновляем pip и очищаем кеш
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade pip

# Копируем все файлы
COPY . .

# Указываем порт
EXPOSE 8000

# Определяем команду запуса приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
