import json
from io import StringIO
import streamlit as st
import requests
import pandas as pd


API_URL = "http://0.0.0.0:8000/api/models"


def validate(uploaded_df, main_df):
    """
        Функция для валидации формата инпут файла
            uploaded_df: pd.DataFrame - загруженный пользователем
            main_df:pd.DataFrame - темплейт с сервера
    """

    if uploaded_df.shape[1] != main_df.shape[1]:
        st.error("Количество колонок не соответствует необходимому")
        return False
    if (uploaded_df.columns != main_df.columns).any():
        st.error("Названия колонок не соответствуют шаблону.")
        return False
    st.success("Загруженный датасет соответствует шаблону!")
    return True


def fetch_example():
    """
        Функция для получения теплейта с сервера
        не требует вводных данных
        """
    response = requests.get(f"{API_URL}/show_example", timeout = 60)
    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.DataFrame(data)[0].str.split(',', expand=True)
        df.columns = df.iloc[0]
        df = df[1:]
        df.columns = df.columns.str.strip()
        return df
    if response.status_code != 200:
        st.error(f"Ошибка при запросе API: {response.status_code}")
    return None


def perform_eda(data):
    """
    Функция для EDA
        params:
            data: pd.DataFrame - загруженный пользователем
    """
    st.header("Анализ данных")
    if st.checkbox("Показать отчет по данным"):
        # Статистика
        st.write('Описание вещественных показателей')
        st.write(data.describe())
        st.write('Описание категориальных показателей')
        st.write(data.describe(include='object'))
        st.write('Плотность данных (Density)')
        st.write(1 - (data.isna().sum()/len(data)).mean())


def params_processing(text_input):
    """
        Функция для получения словаря параметров из текста
            params:
                text_input: str
    """
    params = text_input.split(',')
    hyperparameters = {}

    for param in params:
        if param.strip():  # Пропускаем пустые строки
            name, value = param.split('=')
            name = name.strip()
            value = value.strip()  # Удаляем лишние пробелы

            # Пробуем преобразовать значение в float или bool
            if value == "True":
                hyperparameters[name] = True
            elif value == "False":
                hyperparameters[name] = False
            else:
                try:
                    hyperparameters[name] = float(value)
                except ValueError:
                    hyperparameters[name] = value  # Оставляем как строку

    return hyperparameters


def new_model(data):
    """
        Функция для FIT ручки
            params:
                data: UploadedFile из streamlit
    """
    st.header("Создание модели")
    id_fit = st.text_input('Задайте id для модели:')
    raw_params = st.text_input('Задайте гиперпараметры для модели через '
                               'запятую в формате: '
                               '[имя гиперпараметра] ='
                               '[значение в нужном формате], '
                               'например: alpha = 0.1:')
    hyperparameters = params_processing(raw_params)
    if id_fit != "":
        st.subheader('Статус обучения')
        json_params = {'model_id': id_fit, 'hyperparameters': hyperparameters}
        st.write(f"Полученные на вход параметры: {json_params}")
        files = {
            'data': ('data_sample.csv', data, 'multipart/form-data')
        }

        response = requests.post(f"{API_URL}/fit?jsonfile="
                                 f"{json.dumps(json_params)}",
                                 files=files,
                                 timeout=600)
        if response.status_code == 201:
            st.write('Модель успешно создана и обучена')
        else:
            st.write(response.text)


def load_model():
    """
         Функция для загрузки модели перед обучением
         на входе никакие даннве не передаются
    """
    st.header('Загрузка модели для получения прогнозов')
    id_load = st.text_input('Укажите id модели, которую необходимо'
                            ' загрузить и использовать:')
    if id_load != "":
        ids = {'id': id_load}
        response = requests.post(f"{API_URL}/set_model", json=ids,
                                 timeout=600)
        if response.status_code == 200:
            st.write('Модель загружена и готова к использованию')
        else:
            st.write(response.text)
    return id_load


def predict(id_pred):
    """
        Функция для PREDICT
            params:
                id_pred: str - получен после load_models
    """
    st.header("Предсказание по полученной модели")
    if st.checkbox("Получить предсказание"):
        json_params_pred = {"model_id": str(id_pred)}
        uploaded_pred = st.file_uploader("Загрузите данные в формате "
                                         "CSV-файла, для которых нужно "
                                         "получить предсказание:",
                                         type=["csv"], key=4)
        if uploaded_pred is not None:
            files_pred = {
                'data': ('data_sample.csv', uploaded_pred,
                         'multipart/form-data')
            }
            response = requests.post(f"{API_URL}/predict?jsonfile="
                                     f"{json.dumps(json_params_pred)}",
                                     files=files_pred,
                                     timeout=600)
            content = StringIO(response.content.decode("utf-8"))
            data = pd.read_csv(content).reset_index().drop(columns=['index'])
            if response.status_code == 200:
                st.write('Прогноз получен. Ниже приведен пример из первых строк.')
                st.write('В переменной preds лежат предсказания: '
                         '1 - победа домашней команды, '
                         '0 - проигрыш или ничья')
                st.write(data.head(10))
            else:
                st.write(response.text)


# 5. Просмотр информации о модели и полученных кривых обучения
def model_review():
    """
            Функция для просмотра итоговых моделей
            без дополнительных входных данных
    """
    st.header("Информация о моделях")
    if st.button("Показать информацию о моделях"):
        response = requests.get(f"{API_URL}/show_models", timeout=600)
        if response.status_code == 200:
            st.write(response.json()['message'])
        else:
            st.write(response.json())
            st.error("Ошибка при получении информации о модели.")


def main():
    """
            Функция запуска всех частей приложения
    """
    st.title("Аналитика и прогнозирование исхода футбольных матчей")
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
    template = fetch_example()
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file, sep=",")
        uploaded_file.seek(0)
        validate(uploaded_data, template)
        perform_eda(uploaded_data)
        new_model(uploaded_file)
        id_loaded = load_model()
        predict(id_loaded)
        model_review()


if __name__ == "__main__":
    main()
