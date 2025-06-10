import os
from aiogram import Bot, Dispatcher
from aiogram.utils import executor
from handlers import register_handlers

API_TOKEN = os.getenv('TELEGRAM_API_TOKEN', 'YOUR_TOKEN_HERE')

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

register_handlers(dp)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True) 