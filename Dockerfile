FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Копируем все файлы приложения
COPY . .

# Создаем директорию для логов, если она потребуется
RUN mkdir -p /app/logs

# Экспонируем порт Streamlit
EXPOSE 8501

# Запуск Streamlit-приложения
CMD ["streamlit", "run", "football_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
