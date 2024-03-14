import streamlit as st
import streamlit.components.v1 as components

import time
import io
import json

import pandas as pd
import numpy as np
import plotly_express as px

from openai import OpenAI

from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# initialize session states
if 'keywords' not in st.session_state:
    st.session_state['keywords'] = False
    st.session_state['summary'] = False


st.set_page_config(layout='wide')

# load openai client
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
def save_data_cache(data):
    return data


@st.cache_data
def load_data(path):
    if '.parquet' in path:
        return pd.read_parquet(path)
    if '.csv' in path:
        return pd.read_csv(path)
    

def sort_key(item):
    return not ('about' in item or 'story' in item or 'mission' in item or 'who-we' in item or 'vision' in item)


def url_parse(internal_hrefs):
    normalized_urls = set()
    for url in internal_hrefs:
        parsed_url  = urlparse(url)
        clean_url = urlunparse(parsed_url._replace(fragment=''))
        normalized_urls.add(clean_url)
    normalized_urls = list(normalized_urls)
    return normalized_urls


# @st.cache_resource
def get_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--ignore-certificate-errors')  # Ignore certificate errors
    options.add_argument('--allow-running-insecure-content')  # Allow running insecure content
    options.add_argument('--disable-web-security')  # Bypass CSP; use with caution
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_experimental_option("detach", True)
    return webdriver.Chrome(options=options)


def get_summary(page_text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", 
             "content": "This is the page text scraped off a startup website. Pull out the keywords and a 3-4 sentence summary of what this company does. Format it in json:"},
            {"role": "user", 
             "content": f'Extract from the following page text: [{page_text}]'}
        ]
        )
    return json.loads(completion.choices[0].message.content)


@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.iloc[i : i + rows] for i in range(0, len(input_df), rows)]
    return df


def top_10(df):
  return df.head(10)


def get_cosine_similarity(startups, grants):
    for grant_number in grants.grant_number:
        stacked_embeddings = np.stack(startups['embeddings'])
        grant_embeddings = grants.query('grant_number == @grant_number')['embeddings'].values[0]
        startups[grant_number] = np.dot(stacked_embeddings, grant_embeddings)

    cosine_similarity = startups[['url', 'summary', 'ARPA-H-01',	'ARPA-H-02', 'ARPA-H-03', 'ARPA-H-04', 'ARPA-H-05','ARPA-H-06']]
    return cosine_similarity


# LOAD DATA
nih_keywords_summary = load_data('data/nih_keywords_summary.parquet')
arpa_h = load_data('data/arpa_h.parquet')
startups_data = load_data('data/startups_processed.parquet')
cosine_similarity = get_cosine_similarity(startups_data, arpa_h)

# PROCESS NIH KEY WORDS AND SUMMARY
nih_keywords_summary = nih_keywords_summary.drop_duplicates(subset='abstract_text')
nih_keywords_summary = nih_keywords_summary.drop(columns=['agency_code', 'agency_ic_admin', 'organization', 'agency_ic_fundings', 'arra_funded',
                                                          'mechanism_code_dc', ])
nih_keywords_summary['City'] = nih_keywords_summary['City'].str.upper()
nih_keywords_summary['abstract_text'] = nih_keywords_summary['abstract_text'].str.replace('’', "'")
nih_keywords_summary['keywords'] = nih_keywords_summary['keywords'].apply(lambda lst: [word.replace('’', "'") for word in lst])
nih_keywords_summary['keywords'] = nih_keywords_summary['keywords'].apply(lambda lst: [word.lower() for word in lst])
nih_keywords_summary = nih_keywords_summary.rename(columns={'agency': 'Agency', 'PI_count': 'Principal Investigators Count', 'abstract_text': 'Abstract', 'award_amount': 'Award Amount',
                                                            'appl_id': 'Application ID', 'fiscal_year': 'Fiscal_Year', 'keywords': 'Keywords', 'n_employees': 'Employee Count',
                                                            'opportunity_number': 'Opportunity Number', 'org_name': 'Company', 'phr_text': 'Public Health Relevance',
                                                            'project_detail_url': 'Project URL', 'project_end_date': 'End Date', 'project_start_date': 'Start Date',
                                                            'project_title': 'Project Title', 'publications_count': 'Publications Count', 
                                                            'spending_categories_desc': 'Spending Categories', 'terms': 'Terms', 'summary': 'Summary', 'cfda_code': 'CFDA Code',
                                                            'cong_dist': 'Congressional Dist.', 'direct_cost_amt': 'Direct Costs', 'indirect_cost_amt': 'Indirect Costs', 'city': 'City'
                                                            })


st.markdown('## InsightGrant', unsafe_allow_html=True)

awardee_tab, startups, webscrape, changelog_tab = st.tabs(['Awardees', 'Startups', 'Webscrape' ,'Changelog'])

# Awardee Data
with awardee_tab:

    # FILTER WIDGETS
    keywords_column = st.columns((6, 6))
    with keywords_column[0]:
        keywords = st.text_input('Enter keywords')
        keywords = keywords.split(' ')

    filter_columns = st.columns((3,3,3,3))
    with filter_columns[0]:
        with st.expander('Add/Remove Columns'):
            columns = st.multiselect('Select Columns', options=nih_keywords_summary.columns.sort_values(), default=['Company', 'Summary', 'Keywords'])
    with filter_columns[1]:
        with st.expander('Year(s)'):
            year_options = sorted(list(nih_keywords_summary.Fiscal_Year.unique()))
            years = st.multiselect('Select Year(s)', year_options, default=2023)
    with filter_columns[2]:
        with st.expander('City'):
            city_options = list(nih_keywords_summary.City.unique())
            cities = st.multiselect('Select City', sorted(city_options))

    # MAKE NIH COPY
    temp_df = nih_keywords_summary.copy()

    # FILTER YEARS
    if years:
        years_filter = temp_df['Fiscal_Year'].isin(years)
        temp_df = temp_df[years_filter]

    # FILTER KEYWORDS
    for keyword in keywords:
        keyword_filter = temp_df['Abstract'].str.contains(keyword, case=False)
        temp_df = temp_df[keyword_filter]

    # FILTER CITIES
    if cities:
        with filter_columns[2]:
            city_filter = temp_df['City'].isin(cities)
            temp_df = temp_df[city_filter]

    # FILTER COLUMNS
    temp_df = temp_df[columns]

    main_columns = st.columns((6, 6))

    # DATAFRAME COLUMN
    with main_columns[0]:
        pagination = st.container()
        pagination.data_editor(data=temp_df, use_container_width=True, hide_index=True)

        st.divider()
        
        # AGGREGATIONS AT BOTTOM OF PAGE
        if 'Company' in temp_df.columns:
            agg_columns = st.columns((3,3,5,3))
            agg_df = nih_keywords_summary.loc[temp_df.index]
            agency_counts = (agg_df
                            .drop_duplicates(subset='Company')
                            .groupby(['Agency'])
                            .size()
                            .sort_values(ascending=False)
                            .rename('Companies')
                            )
            location_counts = (agg_df
                            .drop_duplicates(subset='Company')
                            .groupby(['State', 'City'])
                            .size()
                            .sort_values(ascending=False)
                            .rename('Companies')
                            )
            
            with agg_columns[0]:
                st.write(f'Companies: {len(temp_df.Company.unique())}')
                st.write(f'Awards: {len(agg_df["Application ID"].unique())}')
                st.write(f'Average Award: {"${:,.0f}".format(agg_df["Award Amount"].mean())}')
                st.write(f'Median Award: {"${:,.0f}".format(agg_df["Award Amount"].median())}')
            with agg_columns[2]:
                st.write('City Counts')
                st.write(location_counts)
            with agg_columns[1]:
                st.write('Agency Counts')
                st.write(agency_counts)

    # Keyword Rankings
    with main_columns[1]:
        if 'Keywords' in temp_df.columns:

            # st.markdown('## Word Frequency', unsafe_allow_html=True)

            keywords = [word for words in temp_df.Keywords for word in words]
            keywords = pd.Series(keywords)
            st.write((keywords.value_counts() / len(temp_df) * 100).round(1).rename('percentage of abstracts'))


# Startups
with startups:
    columns = st.columns([4, 6, 6])
    with columns[0]:
        grant_number = st.selectbox('select grant', options=arpa_h.grant_number.unique())
        arpa_filtered = arpa_h.query('grant_number == @grant_number')
        st.write(arpa_filtered['grant_description'].values[0])
    with columns[1]:
        temp_df = (cosine_similarity[cosine_similarity[grant_number] > 0.80]
                    .sort_values(by=grant_number, ascending=False)[['url', 'summary']])
        url = st.selectbox('select startup', options=temp_df['url'])
        if url:
            st.write(temp_df.query('url == @url')['summary'].values[0])
            st.write(url)


# WEBSCRAPE
with webscrape:
    # get user url input
    url = st.text_input('Enter url')

    # prepend http if not found in url
    if 'http' not in url:
        url = 'https://' + url

    st.session_state['message'] = st.empty()

    # button for scraping site
    if st.button('Scrape Site'):

        st.session_state['message'].text(f'Initializing Scraper...')

        driver = get_driver()
        driver.get(url)

        st.session_state['message'].text(f"Loading {url}...")

        # parse url
        time.sleep(2)
        url = driver.current_url
        # st.write(url)
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        # st.write(hostname)

        # Use BeautifulSoup to parse and scrape
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        page_text = soup.get_text()

        # get anchor tags
        anchor_tags = driver.find_elements(By.TAG_NAME, "a")

        # Extract href attributes
        hrefs = [tag.get_attribute('href') for tag in anchor_tags if tag.get_attribute('href')]
        
        # drop duplicate hrefs
        hrefs = set(hrefs)

        # create list of internal hrefs
        internal_list = [f'{url}', hostname.split('.')[0]]
        internal_hrefs = [href for href in hrefs if all(include in href for include in internal_list)]
        internal_hrefs = url_parse(internal_hrefs)
        internal_hrefs = set(internal_hrefs)
        internal_hrefs = list(internal_hrefs)
        internal_hrefs = [href for href in internal_hrefs if 'about' in href or 'story' in href or 'mission' in href or 'who-we' in href or 'vision' in href]
        # internal_hrefs = sorted(internal_hrefs, key=sort_key)


        for url in internal_hrefs:
            st.session_state['message'].text(f"Loading {url}...")
            driver.get(url)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            page_text += soup.get_text()

        # Close the browser
        driver.quit()


        st.session_state['message'].text(f"Extracting Keywords/Summary...")
        data = get_summary(page_text)

        st.session_state['keywords'] = data['keywords']
        st.session_state['summary'] = data['summary']
    
    if st.session_state['keywords']:
        webscrape_columns = st.columns((4,4,4))
        with webscrape_columns[0]:
            st.markdown('### Summary:', unsafe_allow_html=True)
            st.write(st.session_state['summary'])
        with webscrape_columns[1]:
            st.markdown('### Keywords:', unsafe_allow_html=True)
            st.write(st.session_state['keywords'])
    
        st.session_state['message'].empty()


# CHANGE LOG
with changelog_tab:
    st.markdown('#### Changelog', unsafe_allow_html=True)
    st.markdown("""
            ##### 12/7/2023:
                - Added 2018-2022 data (~10,000 rows)
                - More columns added (award amount, direct/indirect cost, agency)
                - Cleaned up the keywords column
    """, unsafe_allow_html=True)
    st.markdown("""
            ##### 12/8/2023:
                - Added filter for Years (default is 2023)
                - Added Count for Companies that fit keyword criteria
                - Added Aggregations
                - Added Tabs 
                - Removed pagination from data frame (for now)
    """, unsafe_allow_html=True)
    st.markdown("""
            ##### 12/12/2023:
                - Added Webscraping Feature
    """, unsafe_allow_html=True)
    st.markdown("""
            #### 01/23/2024
                - Added Startup summaries
                - Added Arpa-H summaries
                - Add Matching tool between startup summaries and arpa-h sumamries
    """, unsafe_allow_html=True)
    st.markdown("""
            #### 03/14/2024
                - Changed webscrape model to gpt-3.5-turbo-0125
    """, unsafe_allow_html=True)


    



# st.write('GRID OPTION BUILDER')

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