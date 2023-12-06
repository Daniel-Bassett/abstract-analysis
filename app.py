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
    

@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

st.markdown('## Abstract Search', unsafe_allow_html=True)

keywords = st.text_input('Enter keywords')
keywords = keywords.split(' ')

nih_keywords_summary = load_data('data/nih_keywords_summary.parquet')
nih_keywords_summary = nih_keywords_summary.drop_duplicates(subset='abstract_text')
nih_keywords_summary = nih_keywords_summary[['org_name', 'summary', 'keywords', 'abstract_text']]
nih_keywords_summary['abstract_text'] = nih_keywords_summary['abstract_text'].str.replace('’', "'")

temp_df = nih_keywords_summary.copy()
for keyword in keywords:
    keyword_filter = temp_df['abstract_text'].str.contains(keyword, case=False)
    temp_df = temp_df[keyword_filter]
temp_df = temp_df[['org_name', 'summary', 'keywords']].reset_index(drop=True)

# top_menu = st.columns(3)

# with top_menu[1]:
#     sort_field = st.selectbox("Sort By", options=temp_df.columns)
# with top_menu[2]:
#     sort_direction = st.radio(
#         "Direction", options=["⬆️", "⬇️"], horizontal=True
#     )

# st.data_editor(
#     temp_df,
#     hide_index=True
# )

pagination = st.container()

bottom_menu = st.columns((3, 1, 1))
with bottom_menu[2]:
    batch_size = st.selectbox("Page Size", options=[25, 50, 100])
with bottom_menu[1]:
    total_pages = (
        int(len(temp_df) / batch_size) if int(len(temp_df) / batch_size) > 0 else 1
    )
    current_page = st.number_input(
        "Page", min_value=1, max_value=total_pages, step=1
    )
with bottom_menu[0]:
    st.markdown(f"Page **{current_page}** of **{total_pages}** ")

pages = split_frame(temp_df, batch_size)
pagination.dataframe(data=pages[current_page - 1], use_container_width=True)


# Keyword Rankings

st.divider()

st.markdown('## Keyword Ranking', unsafe_allow_html=True)

keywords = [word for words in nih_keywords_summary.keywords for word in words]
keywords = pd.Series(keywords)
st.write((keywords.value_counts() / len(nih_keywords_summary) * 100).round(2).rename('percentage of abstracts'))