from aiogram import types
from aiogram.dispatcher import Dispatcher
from predictor import make_prediction
from utils import get_logo_path
from teaminfo import get_team_info, get_top_teams


def register_handlers(dp: Dispatcher):
    @dp.message_handler(commands=['start', 'help'])
    async def send_welcome(message: types.Message):
        await message.reply("Привет! Отправь /predict <название_команды>, чтобы получить предсказание результата матча.")

    @dp.message_handler(commands=['predict'])
    async def predict_match(message: types.Message):
        args = message.get_args()
        if not args:
            await message.reply("Пожалуйста, укажи название команды после команды /predict.")
            return
        team_name = args.strip()
        prediction = make_prediction(team_name)
        logo_path = get_logo_path(team_name)
        if logo_path:
            with open(logo_path, 'rb') as photo:
                await message.reply_photo(photo, caption=f"Предсказание для команды {team_name}: {prediction}")
        else:
            await message.reply(f"Предсказание для команды {team_name}: {prediction}\n(Логотип не найден)")

    @dp.message_handler(commands=['teaminfo'])
    async def team_info(message: types.Message):
        args = message.get_args()
        if not args:
            await message.reply("Пожалуйста, укажи название команды после команды /teaminfo.")
            return
        team_name = args.strip()
        info = get_team_info(team_name)
        logo_path = get_logo_path(team_name)
        if info:
            text = '\n'.join([f"{k}: {v}" for k, v in info.items()])
            if logo_path:
                with open(logo_path, 'rb') as photo:
                    await message.reply_photo(photo, caption=text)
            else:
                await message.reply(text + "\n(Логотип не найден)")
        else:
            await message.reply("Информация о команде не найдена.")

    @dp.message_handler(commands=['topteams'])
    async def top_teams(message: types.Message):
        top = get_top_teams(5)
        if not top:
            await message.reply("Не удалось получить топ команд.")
            return
        for team in top:
            text = f"Команда: {team['Команда']}\nСтрана: {team['Страна']}\nЛига: {team['Лига']}\nСредняя стоимость игроков: {team['Средняя стоимость игроков']}"
            logo_path = get_logo_path(team['Команда'])
            if logo_path:
                with open(logo_path, 'rb') as photo:
                    await message.reply_photo(photo, caption=text)
            else:
                await message.reply(text + "\n(Логотип не найден)") 