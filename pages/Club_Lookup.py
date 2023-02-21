
# todo:
# 1. put club lookup into bridgestats.py (and elsewhere?) to validate club entries.

import streamlit as st
import pathlib
import pyarrow.parquet as pq
#import duckdb
import pandas as pd
#import matplotlib.pyplot as plt
import altair as alt
import time
import streamlitlib

@st.experimental_singleton()
def load_club_df(filename):
    acbl_club_df = pd.read_parquet(filename)
    return acbl_club_df

#st.set_page_config(layout="wide", initial_sidebar_state="expanded")
#streamlitlib.widen_scrollbars()

st.header("Lookup Club Information")
st.sidebar.header("Settings for Club Lookup")

st.sidebar.header("Settings")

rootPath = pathlib.Path('.')
acblPath = rootPath.joinpath('.')

acbl_club_dict_filename = 'acbl_clubs.parquet'
acbl_club_dict_file = acblPath.joinpath(acbl_club_dict_filename)
with st.spinner(text="Reading data ..."):
    start_time = time.time()
    acbl_club_df = load_club_df(acbl_club_dict_file)
    end_time = time.time()
    st.caption(f"Data read completed in {round(end_time-start_time,2)} seconds.")

# clubs can be: [] [108571] FTBC, [267096] FLQT, [204891] HH
# todo: need to verify club numbers. At least check for 6 digits.
clubs = st.sidebar.text_input('Narrow search to these ACBL club numbers. Enter one or more 6 digit numbers (empty means all):', placeholder='Enter list of ACBL club numbers', key='Club_Lookup-Club', help='Example: 108571 (Fort Lauderdale Bridge Club')
clubs_l = clubs.replace(',',' ').split()
clubs_regex = '|'.join(clubs_l)

club_names = st.sidebar.text_input('Narrow search to these club names. Enter one or more club names (empty means all):', placeholder='Enter list of club names', key='Club_Lookup-Club_Name')
club_names_l = club_names.split()
club_names_regex = '|'.join(club_names_l)

selected_df = acbl_club_df.convert_dtypes()
selected_df = selected_df if len(clubs_regex)==0 else selected_df[selected_df['id'].astype('string').str.contains(clubs_regex,regex=True)]
selected_df = selected_df if len(club_names_regex)==0 else selected_df[selected_df['name'].str.contains(club_names_regex,case=False,regex=True)]

table, charts = st.tabs(["Data Table", "Charts"])

st.caption(f"Database has {len(acbl_club_df)} rows. {len(selected_df)} rows selected.")
if len(selected_df)==0:
    st.warning('No rows selected')
    st.stop()

with table:
    with st.spinner(text="Creating data table ..."):
        start_time = time.time()
        streamlitlib.ShowDataFrameTable(selected_df)
        end_time = time.time()
        st.caption(f"Data table created in {round(end_time-start_time,2)} seconds.")

with charts:
    pass