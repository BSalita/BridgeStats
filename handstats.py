
# todo:
# 1. move Date to first column or so.
# 2. data table columns need to be ordered by importance. possibly eliminate some unimportant columns.

import streamlit as st
import pathlib
import pickle
import pyarrow.parquet as pq
import duckdb
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import time
import streamlitlib
import bridgestatslib


def create_query(database_name, brs_regex, sample_size=100000):
    query_select = "SELECT *"
    query_from = f"FROM {database_name}"
    query_brs_regex = '' if len(brs_regex)==0 else f"regexp_matches(board_record_string, '{brs_regex}')"
    query_where_string = ' AND'.join(s for s in [query_brs_regex] if len(s))
    query_where = '' if len(query_where_string)==0 else 'WHERE '+query_where_string
    query_sample = f"USING SAMPLE {sample_size}"
    query_limit = '' #if limit==0 else f"LIMIT {limit}"
    query = ' '.join([query_select, query_from, query_where, query_sample, query_limit])
    return query

 
def Stats(club_or_tournament, pair_or_player, chart_options, groupby):

    #st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    #streamlitlib.widen_scrollbars()

    st.header(f"Hand Record Statistics for ACBL {club_or_tournament.capitalize()} Pair Games")
    st.sidebar.header("Settings for Hand Record Statistics")

    key_prefix = club_or_tournament # pair_or_player isn't used here

    rootPath = pathlib.Path('.')
    acblPath = rootPath.joinpath('.')

    acbl_hand_records_augmented_filename = f"acbl_{club_or_tournament}_hand_records_augmented_narrow.parquet"

    # tournament data is as early as 2013? club data as early as 2019?
    start_date = st.sidebar.text_input('Enter start date:', value='2000-01-01', key=key_prefix+'_HandRecord-Start_Date', help='Enter starting date in YYYY-MM-DD format. Earliest year is 2019')
    end_date_default = time.strftime("%Y-%m-%d")
    end_date = st.sidebar.text_input('Enter end date:', value=end_date_default, key=key_prefix+'_HandRecord-End_Date', help='Enter ending date in YYYY-MM-DD format.')

    chart_options = ['Par_Score','CT_NS,CT_EW','DD_N_C,DD_N_D,DD_N_H,DD_N_S,DD_N_N','SL_N_C,SL_N_D,SL_N_H,SL_N_S','SL_N_ML_SJ','HCP_NS,HCP_EW','HCP_N,HCP_E,HCP_S,HCP_W','QT_N,QT_E,QT_S,QT_W','QT_NS,QT_EW','DP_N','DP_N_C,DP_N_D,DP_N_H,DP_N_S','DP_NS,DP_EW','LoTT_Tricks','LoTT_Suit_Length','LoTT_Variance','HCP_NS,DP_NS,DD_N_N','HCP_NS,QT_NS,DD_N_N']
    selected_charts = st.sidebar.multiselect('Select charts to display', chart_options, default=chart_options, key=key_prefix+'_HandRecord-Charts')

    st.sidebar.header("Advanced Settings")

    # select using regex
    brs_regex = st.sidebar.text_input('Restrict results to boards matching this regex:', value='', key=key_prefix+'_HandRecord-brs', help='Example: ^SAK.*$').strip() # e.g. ^SAK.*$

    #sample_percentage = st.sidebar.number_input('Enter sample size percentage (default 1):', value=1., min_value=.001, max_value=100.)
    #sample_percentage = 1.
    table_display_limit = 100 # streamlit gets choked up pretty quickly. need to limit table to 100. todo: check if this has chatnged.
    sample_size = 100000

    st.warning('Table and charts take up to 30 to 60 seconds to render.')

    with st.spinner(text="Reading data ..."):
        start_time = time.time()
        database_name = 'hand_records_arrow'
        if club_or_tournament == 'club':
            hand_records_arrow = bridgestatslib.load_club_hand_records(acbl_hand_records_augmented_filename)
        else:
            hand_records_arrow = bridgestatslib.load_tournament_hand_records(acbl_hand_records_augmented_filename)
        hand_records_len = hand_records_arrow.num_rows
        database_column_names = hand_records_arrow.column_names
        end_time = time.time()
        st.info(f"Data read completed in {round(end_time-start_time,2)} seconds. {hand_records_len} rows read.")

    with st.spinner(text="Selecting database rows ..."):
        start_time = time.time()
        query = create_query(acbl_hand_records_augmented_filename, brs_regex, sample_size) # 'arrow_table', 
        query = st.text_input('Sql query',value=query,label_visibility='hidden', key=key_prefix+'_HandRecord-query') # either use initial query or let user change query
        selected_df = duckdb.arrow(hand_records_arrow).query('hand_records', query).to_df() # todo: can this be cached?
        selected_df = selected_df.drop_duplicates(subset="board_record_string") # using only unique hands (unique board_record_string) otherwise the dups will be double counted.
        uniques = len(selected_df)
        #selected_df = selected_df.sample(sample_size) # need to sample because of limited memory and computation overhead for tables and charts.
        # todo: put this filtering into create_query?
        selected_df = selected_df[selected_df['Date'].between(start_date,end_date)]
        selected_df_len = len(selected_df)
        end_time = time.time()
        st.info(f"Query completed in {round(end_time-start_time,2)} seconds. Database has {hand_records_len} rows. Sampling {sample_size} random rows.  {uniques} unique hands found. {selected_df_len} rows selected and aggregated.")

    # prepare data columns
    selected_df.drop([col for col in selected_df if col.startswith('__')],axis='columns',inplace=True) # remove cols beginning with '__'
    for col in selected_df.select_dtypes('float'): # rounding or {:,.2f} only works on float64!
        selected_df[col] = selected_df[col].astype('float64').round(2)

    #print(list(selected_df.columns))

    table, chart = st.tabs(["Data Table", "Charts"])

    with table:

        with st.spinner(text="Creating data table ..."):
            start_time = time.time()
            #table_df = selected_df.sample(table_display_limit)duckdb.query(f"SELECT * FROM selected_df USING SAMPLE {table_display_limit} ORDER BY {sort_column}").to_df().sort_index() # todo: can this be cached?
            table_df = selected_df.sample(table_display_limit).sort_index()
            st.text(f"Table of Hand Records. {selected_df_len} random rows selected. Table display limited to {len(table_df)} random rows.")
            streamlitlib.ShowDataFrameTable(table_df)
            del table_df
            end_time = time.time()
            st.info(f"Data table created in {round(end_time-start_time,2)} seconds.")

    with chart:

        with st.spinner(text="Creating Charts"):
            start_time = time.time()

            # using st.write because it display in a suitable font, especially font size.
            st.write(f"Abbreviations for Chart Type: CT is Contract Type (passed-out, partial, game, small slam, grand slam), DD is Double Dummy, DP is Distribution Points, HCP is High Card Points, LoTT is Law of Total Tricks, QT is Quick Tricks, SL is Suit Length")
            st.write(f"Abbreviations for individual directions: N is North, S is South, E is East W is West.")
            st.write(f"Abbreviations for pair direction: NS is North-South, EW is East-West.")
            st.write(f"Abbreviations for strains (suits): C is Clubs, D is Diamonds, H is Hearts, S is Spades, N is No-Trump")
            st.write(f"For example: DD_N_N is Double Dummy - North - No-Trump")

            st.text(f"{selected_df_len} random rows selected.")
            bridgestatslib.ShowCharts(selected_df,selected_charts)
            
            end_time = time.time()
            st.info(f"Charts created in {round(end_time-start_time,2)} seconds.")
