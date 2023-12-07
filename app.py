import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder

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
    

# @st.cache_data(show_spinner=False)
# def split_frame(input_df, rows):
#     df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
#     return df

@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.iloc[i : i + rows] for i in range(0, len(input_df), rows)]
    return df


st.markdown('## InsightGrant', unsafe_allow_html=True)


nih_keywords_summary = load_data('data/nih_keywords_summary.parquet')
nih_keywords_summary = nih_keywords_summary.drop_duplicates(subset='abstract_text')
nih_keywords_summary = nih_keywords_summary.drop(columns=['agency_code', 'agency_ic_admin', 'organization', 'agency_ic_fundings', 'arra_funded',
                                                          'mechanism_code_dc', ])
nih_keywords_summary['abstract_text'] = nih_keywords_summary['abstract_text'].str.replace('’', "'")
nih_keywords_summary['keywords'] = nih_keywords_summary['keywords'].apply(lambda lst: [word.replace('’', "'") for word in lst])
nih_keywords_summary['keywords'] = nih_keywords_summary['keywords'].apply(lambda lst: [word.lower() for word in lst])
nih_keywords_summary = nih_keywords_summary.rename(columns={'PI_count': 'Principal Investigators Count', 'abstract_text': 'Abstract', 'award_amount': 'Award Amount',
                                                            'appl_id': 'Application ID', 'fiscal_year': 'Fiscal Year', 'keywords': 'Keywords', 'n_employees': 'Employee Count',
                                                            'opportunity_number': 'Opportunity Number', 'org_name': 'Company', 'phr_text': 'Public Health Relevance',
                                                            'project_detail_url': 'Project URL', 'project_end_date': 'End Date', 'project_start_date': 'Start Date',
                                                            'project_title': 'Project Title', 'publications_count': 'Publications Count', 
                                                            'spending_categories_desc': 'Spending Categories', 'terms': 'Terms', 'summary': 'Summary', 'cfda_code': 'CFDA Code',
                                                            'cong_dist': 'Congressional Dist.', 'direct_cost_amt': 'Direct Costs', 'indirect_cost_amt': 'Indirect Costs'
                                                            })


keywords_column = st.columns((6, 6))
with keywords_column[0]:
    keywords = st.text_input('Enter keywords')
    keywords = keywords.split(' ')

columns_column = st.columns((6,6))
with columns_column[0]:
    with st.expander('Add/Remove Columns'):
        columns = st.multiselect('Select Columns', options=nih_keywords_summary.columns.sort_values(), default=['Company', 'Summary', 'Keywords'])


# FILTER KEYWORDS
temp_df = nih_keywords_summary.copy()
for keyword in keywords:
    keyword_filter = temp_df['Abstract'].str.contains(keyword, case=False)
    temp_df = temp_df[keyword_filter]
temp_df = temp_df[columns]



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
    pagination.data_editor(data=pages[current_page - 1], use_container_width=True, hide_index=True, column_config={"Select": st.column_config.CheckboxColumn(required=True)})


with main_columns[1]:
# Keyword Rankings
    if 'Keywords' in temp_df.columns:

        # st.markdown('## Word Frequency', unsafe_allow_html=True)

        keywords = [word for words in temp_df.Keywords for word in words]
        keywords = pd.Series(keywords)
        st.write((keywords.value_counts() / len(temp_df) * 100).round(1).rename('percentage of abstracts'))


st.markdown('#### Change Log', unsafe_allow_html=True)
st.markdown("""
        ##### 12/7/2023:
            - Added 2018-2022 data (~10,000 rows)
            - More columns added (award amount, direct/indirect cost, agency)
""", unsafe_allow_html=True)


# st.write('TEST')

# def dataframe_with_selections(df):
#     df_with_selections = df.copy()
#     df_with_selections.insert(0, "Select", False)

#     # Get dataframe row-selections from user with st.data_editor
#     edited_df = st.data_editor(
#         df_with_selections,
#         hide_index=True,
#         column_config={"Select": st.column_config.CheckboxColumn(required=True)},
#         disabled=df.columns,
#     )

#     # Filter the dataframe using the temporary column, then drop the column
#     selected_rows = edited_df[edited_df.Select]
#     return selected_rows.drop('Select', axis=1)


# selection = dataframe_with_selections(temp_df)
# st.write("Your selection:")
# st.write(selection)

# for index, row in selection.iterrows():
#     st.write(f"""
#             Company: {row['Company']}
#             Summary: {row['Summary']}
# """, unsafe_allow_html=True)
#     st.divider()

# gd = GridOptionsBuilder.from_dataframe(temp_df)
# gd.configure_pagination(enabled=True, paginationPageSize=25)
# gridoptions = gd.build()
# AgGrid(temp_df, gridOptions=gridoptions)