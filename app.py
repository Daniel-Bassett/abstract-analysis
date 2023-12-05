import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

import time
import io

import pandas as pd
import plotly_express as px

from openai import OpenAI

try:
    with open('../../openai_key.txt', 'r') as file:
        API_KEY = file.read().strip()
except:
    pass

try:
    client = OpenAI(api_key=API_KEY)
except:
    client = OpenAI(api_key=st.secrets["api_key"])


@st.cache_data
def load_data(path):
    if '.parquet' in path:
        return pd.read_parquet(path)
    if '.csv' in path:
        return pd.read_csv(path)


keywords = st.text_input('Enter keywords')
keywords = keywords.split(' ')

nih_first_100 = load_data('data/nih_first_100.parquet')
nih_first_100 = nih_first_100[['org_name', 'summary', 'keywords', 'abstract_text']]

temp_df = nih_first_100.copy()
for keyword in keywords:
    keyword_filter = temp_df['abstract_text'].str.contains(keyword, case=False)
    temp_df = temp_df[keyword_filter]

st.data_editor(
    temp_df,
    hide_index=True
)










# st.header("test html import")

# HtmlFile = open("data/country.html", 'r', encoding='utf-8')
# source_code = HtmlFile.read() 
# print(source_code)
# components.html(source_code, height=1000, width=1300)