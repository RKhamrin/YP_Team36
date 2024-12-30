import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict
from api.api_route import router
import logging
from logging.handlers import RotatingFileHandler
import os

# Создание директории для логов
if not os.path.exists('logs'):
    os.makedirs('logs')

# Настройка логирования с ротацией
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_file = "logs/app.log"
handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
handler.setFormatter(log_formatter)
handler.setLevel(logging.DEBUG)

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Получен запрос: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Ответ: {response.status_code}")
    return response

@app.get("/")
async def root():
    """Функция получения статуса сервиса"""
    logger.info("Запрос на проверку статуса приложения")
    return StatusResponse(status='App healthy')

# Подключение маршрутов
app.include_router(router)

if __name__ == "__main__":
    logger.info("Запуск приложения model_trainer")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
