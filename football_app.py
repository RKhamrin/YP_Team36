import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from sklearn.model_selection import train_test_split

#if not os.path.exists('logs'):
#    os.makedirs('logs')

#logging.basicConfig(
#    filename='logs/app.log',
#    level=logging.INFO,
#    format='%(asctime)s - %(levelname)s - %(message)s'
#)

template_data = pd.read_csv("teams_matches_stats-2.csv").head(1)

def validate(uploaded_df, main_df):
    if uploaded_df.shape[1] != main_df.shape[1]:
        st.error("Количество колонок не соответствует необходимому")
        return False
    if not all(uploaded_df.columns == main_df.columns):
        st.error("Названия колонок не соответствуют шаблону.")
        return False
    for col in main_df.columns:
        if uploaded_df[col].dtype != main_df[col].dtype:
            st.error(f"Тип данных для колонки '{col}' не соответствует шаблону.")
            return False
    st.success("Загруженный датасет соответствует шаблону!")
    return True


st.title("Аналитика и прогнозирование исхода футбольных матчей")

st.header("Шаг 1: Загрузка данных")
uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    validate(uploaded_data, template_data)