import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

import time
import io

import pandas as pd
import plotly_express as px

from openai import OpenAI

st.set_page_config(layout='wide')

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


nih_keywords_summary = load_data('data/nih_keywords_summary.parquet')
nih_keywords_summary = nih_keywords_summary.drop_duplicates(subset='abstract_text')
nih_keywords_summary['abstract_text'] = nih_keywords_summary['abstract_text'].str.replace('’', "'")
nih_keywords_summary = nih_keywords_summary.rename(columns={'PI_count': 'Principal Investigators Count', 'abstract_text': 'Abstract',
                                                            'appl_id': 'Application ID', 'fiscal_year': 'Fiscal Year', 'keywords': 'Keywords', 'n_employees': 'Employee Count',
                                                            'opportunity_number': 'Opportunity Number', 'org_name': 'Company', 'phr_text': 'Public Health Relevance',
                                                            'project_detail_url': 'Project URL', 'project_end_date': 'End Date', 'project_start_date': 'Start Date',
                                                            'project_title': 'Project Title', 'publications_count': 'Publications Count', 
                                                            'spending_categories_desc': 'Spending Categories', 'terms': 'Terms', 'summary': 'Summary'})


keywords_column = st.columns((6, 6))
with keywords_column[0]:
    keywords = st.text_input('Enter keywords')
    keywords = keywords.split(' ')

columns_column = st.columns((6,6))
with columns_column[0]:
    with st.expander('Add/Remove Columns'):
        columns = st.multiselect('Select Columns', options=nih_keywords_summary.columns.sort_values(), default=['Company', 'Summary', 'Keywords'])

# with columns_column[1]:
#     with st.expander('Column Explanations'):
#         st.write("""
#                     1. Test
#                     2. Test
#                 """)

# nih_keywords_summary = nih_keywords_summary[['org_name', 'summary', 'keywords', 'abstract_text']]
# nih_keywords_summary = nih_keywords_summary[['org_name', 'summary', 'keywords', 'abstract_text']]




temp_df = nih_keywords_summary.copy()
for keyword in keywords:
    keyword_filter = temp_df['Abstract'].str.contains(keyword, case=False)
    temp_df = temp_df[keyword_filter]
temp_df = temp_df[columns].reset_index(drop=True)

# top_menu = st.columns(3)

# with top_menu[1]:
#     sort_field = st.selectbox("Sort By", options=temp_df.columns)
# with top_menu[2]:
#     sort_direction = st.radio(
#         "Direction", options=["⬆️", "⬇️"], horizontal=True
#     )

main_columns = st.columns((6, 6))

# DATAFRAME COLUMN
with main_columns[0]:
    pagination = st.container()

    bottom_menu = st.columns((3, 5, 4))
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
    pagination.data_editor(data=pages[current_page - 1], use_container_width=True, hide_index=True)


with main_columns[1]:
# Keyword Rankings
    if 'Keywords' in temp_df.columns:

        # st.markdown('## Word Frequency', unsafe_allow_html=True)

        keywords = [word for words in temp_df.Keywords for word in words]
        keywords = pd.Series(keywords)
        st.write((keywords.value_counts() / len(temp_df) * 100).round(1).rename('percentage of abstracts'))