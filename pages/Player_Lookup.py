# todo:
# 1. Replace Club, and Player entry with delux code in bridgestats.

import streamlit as st
import pathlib
import pyarrow.parquet as pq
#import duckdb
import pandas as pd
#import matplotlib.pyplot as plt
import altair as alt
import time
import bridgestatslib
import sys
sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
import streamlitlib # must be placed after sys.path.append. vscode re-format likes to move this to the top
import polars as pl


#st.set_page_config(layout="wide", initial_sidebar_state="expanded")
#streamlitlib.widen_scrollbars()

st.header("Lookup Player Information")
st.sidebar.header("Settings for Player Lookup")

st.sidebar.header("Settings")

key_prefix = 'Player_Lookup'

rootPath = pathlib.Path('.')
dataPath = rootPath.joinpath('data')

with st.spinner(text="Reading data ..."):
    start_time = time.time()
    acbl_player_name_dict_filename = 'acbl_player_info.parquet'
    acbl_player_name_dict_file = dataPath.joinpath(acbl_player_name_dict_filename)
    acbl_player_df = bridgestatslib.load_player_info_df(acbl_player_name_dict_file)
    end_time = time.time()
    st.caption(f"Data read completed in {round(end_time-start_time,2)} seconds. {acbl_player_df.height} rows read.")

# clubs can be: [] [108571] FTBC, [267096] FLQT, [204891] HH
clubs = st.sidebar.text_input('Narrow search to these ACBL club numbers. Enter one or more 6 digit ACBL club numbers (empty means all):', placeholder='Enter list of ACBL club numbers', key=key_prefix+'-Club', help='Example: 108571 (Fort Lauderdale Bridge Club')
clubs_l = clubs.replace(',',' ').split()
clubs_regex = '|'.join(clubs_l)

# todo: need to verify player numbers. At least check for 7 digits.
# 2663279 (Robert Salita), 2454602 (Kerry Flom), 6811434 (Mark Itabashi), 1709925 (Neil Silverman) 9173862 (Brad Moss)
player_numbers = st.sidebar.text_input('Narrow search to these ACBL player numbers. Enter one or more 7 digit numbers (empty means all):', placeholder='Enter list of ACBL numbers', key=key_prefix+'-ACBL_number')
player_numbers_l = player_numbers.replace(',',' ').split()
player_numbers_regex = '|'.join(player_numbers_l)

player_names = st.sidebar.text_input('Narrow search to these last names. Enter one or more last names (empty means all):', placeholder='Enter list of last names', key=key_prefix+'-Last_Name')
player_names_l = player_names.split()
player_names_regex = '|'.join(player_names_l)

# Cast columns to proper types and apply filters
selected_df = acbl_player_df.with_columns([
    pl.col('mp_total').cast(pl.Float64),
    pl.col('club').cast(pl.Utf8),
    pl.col('acbl_number').cast(pl.Utf8)
])

selected_df = selected_df if len(clubs_regex)==0 else selected_df.filter(pl.col('club').str.contains(clubs_regex))
selected_df = selected_df if len(player_numbers_regex)==0 else selected_df.filter(pl.col('acbl_number').str.contains(player_numbers_regex))
selected_df = selected_df if len(player_names_regex)==0 else selected_df.filter(pl.col('last_name').str.contains(player_names_regex, ignore_case=True))

# Drop master point columns per ACBL privacy requirements
selected_df = selected_df.drop(selected_df.select(pl.col("^mp_.*$")).columns)

table, charts = st.tabs(["Data Table", "Charts"])

st.caption(f"Database has {acbl_player_df.height} rows. {selected_df.height} rows selected.")

if selected_df.height == 0:
    st.warning('No rows selected')
    st.stop()

with table:
    with st.spinner(text="Creating data table ..."):
        start_time = time.time()
        streamlitlib.ShowDataFrameTable(selected_df)
        end_time = time.time()
        st.caption(f"Data table created in {round(end_time-start_time,2)} seconds.")

with charts:
    with st.spinner(text="Creating charts ..."):
        start_time = time.time()
        
        # Convert to pandas for Altair
        chart_df = selected_df.to_pandas()
        
        col = 'rank_description'
        x = f'count({col}):Q'
        y = f'{col}:N'
        title_x = 'Count'
        title_y = col.replace('_',' ').title()
        title = f'Frequency of {title_y}'
        sort_y = '-x'
        c = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(alt.X(x, title=title_x),alt.Y(y, title=title_y, sort=sort_y))
            .properties(width=1000, height=500,title=title)
            .configure_axis(labelFontSize=14, titleFontSize=20, labelLimit=200, titleFontWeight='bold')
            .configure_title(fontSize=20, offset=5, orient='top', anchor='middle')
        )
        st.altair_chart(c)
                
        end_time = time.time()
        st.caption(f"Charts created in {round(end_time-start_time,2)} seconds.")
