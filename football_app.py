import streamlit
import streamlit as st
import requests
import pandas as pd
import json
import uvicorn
import matplotlib.pyplot as plt
import logging
import os
from sklearn.model_selection import train_test_split
from io import StringIO

#if not os.path.exists('logs'):
#    os.makedirs('logs')

#logging.basicConfig(
#    filename='logs/app.log',
#    level=logging.INFO,
#    format='%(asctime)s - %(levelname)s - %(message)s'
#)
API_URL = "http://0.0.0.0:8000/api/models"
#template_data = pd.read_csv("teams_matches_stats-2.csv").head(1)

def validate(uploaded_df, main_df):
    if uploaded_df.shape[1] != main_df.shape[1]:
        st.error("Количество колонок не соответствует необходимому")
        return False
    if not all(uploaded_df.columns == main_df.columns):
        st.error("Названия колонок не соответствуют шаблону.")
        return False
    #for col in main_df.columns:
    #    if uploaded_df[col].dtype != main_df[col].dtype:
    #        st.error(f"Тип данных для колонки '{col}' не соответствует шаблону.")
    #        return False
    st.success("Загруженный датасет соответствует шаблону!")
    return True

def fetch_example():
    response = requests.get(f"{API_URL}/show_example")
    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.DataFrame(data)[0].str.split(',',expand = True)
        df.columns = df.iloc[0]
        df = df[1:]
        df.columns = df.columns.str.strip()
        return df
    else:
        st.error(f"Ошибка при запросе API: {response.status_code}")
        return None

def perform_eda(data):
    st.header("Анализ данных")
    if st.checkbox("Показать отчет по данным"):
    # Статистика
        st.write('Описание вещественных показателей')
        st.write(data.describe())
        st.write('Описание категориальных показателей')
        st.write(data.describe(include = 'object'))
        st.write('Плотность данных (Density)')
        st.write(1 - (data.isna().sum()/len(data)).mean())
    # Визуализация
        #st.subheader("Гистограмма")
        #data.hist(bins=30)
        #plt.show()
        #st.pyplot()
def params_processing(text_input):
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
    st.header("Создание модели")
    id = st.text_input('Задайте id для модели:')
    raw_params = st.text_input('Задайте гиперпараметры для модели через запятую в формате: [имя гиперпараметра]'
                                    '=[значение в нужном формате], например: alpha = 0.1')
    hyperparameters = params_processing(raw_params)
    st.write(hyperparameters)
    if id is not None:
        st.subheader('Обучение модели')
        json_params = {'model_id':id,'hyperparameters': hyperparameters}
        st.write(json_params)
        #csv_data = data.to_csv(index=False)
        #files = {'data': ('data.csv', csv_data, 'text/csv')}
        #headers = {'Content-Type': 'application/json'}
        params = {'data':data, 'jsonfile':json_params}
        response = requests.post(f"{API_URL}/fit", params)

        #response = requests.post(f"{API_URL}/fit", data = data.to_csv(), json = json_params)
        st.write(response.text)


def main():
    st.title("Аналитика и прогнозирование исхода футбольных матчей")
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
    template = fetch_example()
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file, sep = ",", index_col = 0)
        validate(uploaded_data, template)
        perform_eda(uploaded_data)
        new_model(uploaded_file)

if __name__ == "__main__":
    main()

#uvicorn.run("model_trainer.main:app", host="0.0.0.0", port=8000, reload=True)
#streamlit run football_app.py