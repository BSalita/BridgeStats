# todo:
# 1. is match point charting implemented and proper? something's .5%
# 2. move club, player, pair validations to bridgestatslib.
# 3. looks like 2500 hrd contains 2500 hand records with superceded hand record ids. Dropping dups here, keeping latest. But this step should be done in hand_record_clean.
# 4. Output chart label with names instead of Declarer_Pairs
# 5. Due to rendering delays, limit charts to the 100 most frequent x labels.
# 6. Implement column of total of HCP per session per direction. Analyse how players perform when they have more HCP, voids, etc.

import streamlit as st
import pathlib
import re
import pyarrow.parquet as pq
import pandas as pd
import polars as pl
import altair as alt
import matplotlib.pyplot as plt
import time
import bridgestatslib
import sys
import os
import streamlitlib  # assumed import

# todo: doesn't some variation of import chatlib.chatlib work instead of using sys.path.append such as exporting via __init__.py?
#import acbllib.acbllib
#import streamlitlib.streamlitlib
#import chatlib.chatlib
#import mlBridgeLib.mlBridgeLib
sys.path.append(str(pathlib.Path.cwd().joinpath('acbllib')))  # global
#sys.path.append(str(pathlib.Path.cwd().joinpath('chatlib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
# streamlitlib, mlBridgeLib, chatlib must be placed after sys.path.append. vscode re-format likes to move them to the top
import acbllib
#import chatlib  # must be placed after sys.path.append. vscode re-format likes to move this to the top
import mlBridgeLib # must be placed after sys.path.append. vscode re-format likes to move this to the top
import streamlitlib # must be placed after sys.path.append. vscode re-format likes to move this to the top


def create_query(database_name, groupby, having, limit, columns, clubs, players, pairs, min_declares, stat_column, minimum_mps, maximum_mps, start_date, end_date):
    query_select = f"SELECT {columns}"
    query_from = f"FROM {database_name}"
    query_where_clubs = '' if len(clubs) == 0 else f"Club IN ({','.join(clubs)})"
    #query_where_players = '' if len(players) == 0 else f"Declarer IN ({','.join(players)})" # f"NNum IN ({','.join(players)}) OR ENum IN ({','.join(players)}) OR SNum IN ({','.join(players)}) OR WNum IN ({','.join(players)})"
    query_where_players = '' if len(players) == 0 else f"NNum IN ({','.join(players)}) OR ENum IN ({','.join(players)}) OR SNum IN ({','.join(players)}) OR WNum IN ({','.join(players)})"
    # issue is ordering of player numbers within Declarer_Pair. Better to use CONCAT(), swap players within pairs thus doubling, or always have Declarer_Pair sorted in db breaking NS,EW ordering?
    query_where_pairs = '' if len(pairs) == 0 else f"CONCAT(Declarer,'_',Dummy) IN ('"+"','".join(pairs)+"') OR CONCAT(Dummy,'_',Declarer) IN ('"+"','".join(pairs)+"')" # OR Defender_Pair IN ('"+"','".join(pairs)+"')"
    query_where_mps = '' #f"Declarer_MP BETWEEN {minimum_mps} AND {maximum_mps}"
    query_where_dates = f"Date BETWEEN '{start_date}' AND '{end_date}'"
    query_where_string = ' AND '.join(s for s in [
                                      query_where_clubs, query_where_players, query_where_pairs, query_where_mps, query_where_dates] if len(s))
    query_where = '' if len(
        query_where_string) == 0 else 'WHERE '+query_where_string
    query_group = '' if len(groupby) == 0 else f"GROUP BY {groupby}"
    query_having = '' if len(having) == 0 else f"HAVING {having}"
    # Can't use stat_column when it's Player_Detail so just obsolete for now
    query_ordered_by = ''  # f"ORDER BY AVG({stat_column}) DESC"
    query_limit = '' if limit == 0 else f"LIMIT {limit}"
    query = ' '.join([query_select, query_from, query_where,
                     query_group, query_having, query_ordered_by, query_limit])
    return query


def Stats(club_or_tournament, pair_or_player, chart_options, groupby):

    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    streamlitlib.widen_scrollbars()

    st.header(f"{pair_or_player.capitalize()} Statistics for ACBL {club_or_tournament.capitalize()} Pair Games")
    st.sidebar.header(f"Settings for {pair_or_player.capitalize()} Statistics")

    key_prefix = groupby[0]+club_or_tournament+'_'+pair_or_player # assumes groupby[0] is unique

    rootPath = pathlib.Path('.')
    dataPath = rootPath.joinpath('data')

    with st.spinner(text="Reading player data ..."):
        acbl_player_d_filename = f"acbl_{club_or_tournament}_player_name_dict.pkl"
        acbl_player_d_file = dataPath.joinpath(acbl_player_d_filename)
        if club_or_tournament == 'club':
            acbl_player_d = bridgestatslib.load_club_player_d(acbl_player_d_file)
        else:
            acbl_player_d = bridgestatslib.load_tournament_player_d(acbl_player_d_file)

    with st.spinner(text="Reading hand record data ..."):
        acbl_hand_records_d_filename = f"acbl_{club_or_tournament}_hand_records_d.pkl"
        acbl_hand_records_d_file = dataPath.joinpath(acbl_hand_records_d_filename)
        if club_or_tournament == 'club':
            hrd = bridgestatslib.load_club_hand_records_d(acbl_hand_records_d_file)
        else:
            hrd = bridgestatslib.load_tournament_hand_records_d(acbl_hand_records_d_file)

    # Data source for dataframe
    acbl_board_results_augmented_filename = f"acbl_{club_or_tournament}_board_results_augmented.parquet"
    acbl_board_results_augmented_file = dataPath.joinpath(acbl_board_results_augmented_filename)

    # todo: implement verification of club numbers by looking them up in dict? At least check for 6 digits.
    # 108571 is Fort Lauderdale, 267096 is Fort Lauderdale Quick Tricks, 204891 Hilton Head
    if club_or_tournament == 'club':
        # streamlit bug? Must use non-empty value else preserves state between page loads when no 'key' is used.
        #clubs = st.sidebar.selectbox('Club ACBL Numbers - Restrict results to these 6 digit ACBL club numbers (empty means all):', options=['','108571','267096'], key=key_prefix+'-Clubs',help='Enter zero or more ACBL club numbers. Use Club Lookup, in above list, to find a club number.')
        clubs = st.sidebar.text_input('Club ACBL Numbers - Restrict results to these 6 digit ACBL club numbers (empty means all). Examples: 108571  267096', placeholder='Enter club numbers', value='108571 267096', key=key_prefix+'-Clubs',help='Enter zero or more ACBL club numbers. Use Club Lookup, in above list, to find a club number. Examples: 108571  267096')
        clubs = clubs.replace(',', ' ').replace('_', ' ').split()
        clubs = [] if clubs == [''] else clubs

        for club in clubs:
            if not re.match(r'^\d{6}$',club): # implement actual club check - if club not in acbl_clubs:
                st.warning(f"Club {club} has invalid syntax. Expecting one or more six digit ACBL club numbers e.g 123456. Please correct.")
                st.stop()
    else:
        clubs = []

    if pair_or_player == 'player':
        # 2663279 (Robert Salita), 2454602 (Kerry Flom), 6811434 (Mark Itabishi?), 1709925 (Neil Silverman) 2997630 (Bella Ionis-Sorren), 4464109 (Curley Anderson)
        # streamlit bug? Must use non-empty value else preserves state between page loads when no 'key' is used.
        #players = st.sidebar.selectbox('Player ACBL Numbers - Restrict results to these 7 digit ACBL player numbers (empty means all):', options=['','2663279'], key=key_prefix+'-Players',help='Enter zero or more ACBL player numbers. Use Player Lookup, in above list, to find a player number.')
        players = st.sidebar.text_input('Player ACBL Numbers - Restrict results to these 7 digit ACBL player numbers (empty means all). Examples: 2663279 9524304 6941303 6941346', placeholder='Enter player numbers', value='2663279 9524304 6941303 6941346', key=key_prefix+'-Players',help='Enter zero or more ACBL player numbers. Use Player Lookup, in above list, to find a player number. Examples: 2663279 9524304 6941303 6941346')
        players = players.replace(',',' ').replace('_',' ').split()
        players = [] if players == [''] else players

        for player in players:
            if not re.match(r'^\d{7}$',player):
                st.warning(f"Player {player} has invalid syntax. Expecting one or more valid seven digit ACBL player numbers. Please correct.")
                st.stop()
            if player not in acbl_player_d:
                st.warning(f"Player {player} is unknown. Remove from list.")
                st.stop()
    else:
        players = []

    if pair_or_player == 'pair':
        # todo: pairs need to have lowest number acbl number first.
        # 1709925_6811434 (Neil Silverman, Mark Itabashi), 2130335_2342200 (Lee Atkinson and Jack Jones), 2997630_4441389 (Bella, Titus), 2130335_2342200 2997630_4441389
        pairs = st.sidebar.text_input('Pair ACBL Numbers - Restrict results to these pairs. Use two 7 digit ACBL player numbers separated by an underscore (empty means all). Examples: 2663279_9524304  6941303_6941346', placeholder='Enter Pair Numbers', value='2663279_9524304 6941303_6941346', key=key_prefix+'-Pairs') # streamlit bug? Must use non-empty value else preserves state between page loads when no 'key' is used.
        pairs = pairs.replace(',',' ').split()
        pairs = [] if pairs == [''] else pairs

        for pair in pairs:
            if not re.match(r'^\d{7}_\d{7}$',pair):
                st.warning(f"Pair {pair} has invalid syntax. Expecting two valid ACBL numbers separated by an underscore e.g. 1234567_7654321. You can enter multiple sets of pair numbers. Please correct.")
                st.stop()
            for player in pair.split('_'):
                if player not in acbl_player_d:
                    st.warning(f"Player {player} is unknown. Remove from list.")
                    st.stop()
                if player not in players:
                    players.append(player)
        # to rewrite pairs in sorted order: pairs = ['_'.join(sorted(pair.split('_'))) for pair in pairs]
    else:
        pairs = []

    # todo: possible to use chart_options?
    # listing most likely columns to want sorted. not sure what the case is for others.
    sort_options = ['Declarer_Pct']
    sort_options += ['Declarer_Score','Declarer_DD_Score','Declarer_ParScore','Declarer_SD_Score','Declarer_SD_Score_Max']
    sort_options += ['Declarer_DD_GE','Declarer_ParScore_GE','Declarer_Tricks_DD_Diff','Declarer_ParScore_DD_Diff','Declarer_Score_DD_Diff','OverTricks','JustMade','UnderTricks'] #,'Declarer_MP']
    sort_options += ['Declarer_SD_Score_Diff','Declarer_SD_Score_Max_Diff']
    sort_options += ['Declarer_ParScore_Pct','Declarer_SD_Pct','Declarer_SD_Pct_Max','Declarer_SD_Pct_Diff','Declarer_SD_Pct_Max_Diff','Declarer_SD_ParScore_Pct_Diff','Declarer_SD_ParScore_Pct_Max_Diff'] # can't get mean of contracts but can do value_counts() (not implemented).
    stat_column = st.sidebar.selectbox('Sort table by:',options=sort_options+['Count'],key=key_prefix+'-Stat',help='Choose statistic to use as primary sort') # use ['Date']+ if player or pair is specified?

    # stat_column becomes the sort_column
    sort_column = stat_column.strip()

    minimum_declares = 0 if len(players) or len(pairs) else 6 if groupby[0] == 'Session' else 30
    min_declares = st.sidebar.number_input(f"Enter minimum number of times a player must have declared (default {minimum_declares}):",  value=minimum_declares, min_value=0, key=key_prefix+'-Declares-Min')
    
    top_ranked = st.sidebar.number_input('Enter number of top ranked results to show (default 100):', value=100, min_value=10, key=key_prefix+'-Declares-Top-Rank') # depends on stat_column?

    # tournament data is as early as 2013? club data as early as 2019?
    start_date = st.sidebar.text_input('Enter start date:', value='2019-01-01', key=key_prefix+'-Start_Date', help='Enter starting date in YYYY-MM-DD format. Earliest year is 2019')
    end_date_default = time.strftime('%Y-%m-%d')
    end_date = st.sidebar.text_input('Enter end date:', value=end_date_default, key=key_prefix+'-End_Date', help='Enter ending date in YYYY-MM-DD format.')

    minimum_mps_value = 0 if len(players) or len(pairs) else 300
    #minimum_mps = st.sidebar.number_input('Enter pair minimum master points (default is 0):', value=minimum_mps_value, min_value=0, key=key_prefix+'-MP-Min')
    minimum_mps = minimum_mps_value

    # todo: make 9999999 into a variable.
    #maximum_mps = st.sidebar.number_input('Enter maximum master points (default is 9999999):', value=9999999, max_value=9999999, key=key_prefix+'-MP_Max')
    maximum_mps = 9999999
    
    st.warning('Table and charts take from a few seconds to 120 seconds to render. Please be patient. Wait for the man in the upper-right corner to stop running ... Initial load is slowest.')

    with st.spinner(text="Reading board result data ..."):
        start_time = time.time()
        database_name = 'board_results_arrow'
        if club_or_tournament == 'club':
            board_results_arrow = bridgestatslib.load_club_board_results(acbl_board_results_augmented_file)
        else:
            board_results_arrow = bridgestatslib.load_tournament_board_results(acbl_board_results_augmented_file)
        board_results_len = board_results_arrow.select(pl.count()).collect().item()
        database_column_names = board_results_arrow.schema.keys() # todo: polars recommends .collect().schema.keys()
        #st.write(database_column_names)
        end_time = time.time()
        st.info(f"Data read completed in {round(end_time-start_time,2)} seconds. {board_results_len} rows read.")

    # todo: create dict of column name having a list of chart types: {'Tricks':['F']}
    # event data:  'Date', 'Session', 'HandRecord', 'mp_limit'
    # player id data: 'NNum', 'SNum', 'ENum', 'WNum', 'NName', 'SName', 'EName', 'WName', 'Declarer', 'OnLead', 'Dummy', 'NotOnLead', 'Declarer_Name', other names ...
    # Master point data: 'MP_N', 'MP_S', 'MP_E', 'MP_W', 'Declarer_MP', 'Dummy_MP', 'OnLead_MP', 'NotOnLead_MP', 'NS_Geo_MP', 'EW_Geo_MP', 'Declarer_Geo_MP', 'Defender_Geo_MP'
    # hand record board data: 'HandRecordBoard', 'Board', 'Dealer', 'Vul', 'Par_Score'
    # contract data: 'contract', 'BidLvl', 'BidSuit', 'Dbl', 'Declarer_Direction', 'ContractType'
    # board result: 'Tricks', 'Result', 'match_points_NS', 'match_points_EW', 'Score_NS', 'Score_EW', 'Pct_NS', 'Pct_EW'
    # Declarer data: 'Declarer_Score', 'Declarer_ParScore', 'Declarer_Pct', 'Declarer_DD_Tricks', 'Declarer_DD_Score', 'Declarer_DD_Pct', 'Declarer_Tricks_DD_Diff', 'Declarer_Score_DD_Diff', 'Declarer_ParScore_DD_Diff'
    # Pair data: 'Declarer_Pair', 'Defender_Pair', 'pair_number_NS', 'pair_number_EW'
    
    # board_results columns:
    # 'Key', 'Club', 'Date', 'ClubDate', 'Session', 'HandRecord',
    # 'HandRecordBoard', 'Board', 'Pair', 'NNum', 'NName', 'SNum', 'SName',
    # 'ENum', 'EName', 'WNum', 'WName', 'PairNS', 'PairEW', 'MP_N', 'MP_S',
    # 'MP_E', 'MP_W', 'MP_NS', 'MP_EW', 'Score', 'MatchP', 'Pct', 'NSPair',
    # 'EWPair', 'BidLvl', 'BidSuit', 'Dbl', 'Declarer_Direction', 'Tricks', 'Round',
    # 'Table', 'Lead', 'Result', 'Declarer', 'OnLead', 'Dummy', 'NotOnLead',
    # 'HandRecordBoardScore', 'ContractType', 'Dealer', 'Par_Score',
    # 'Declarer_MP', 'Dummy_MP', ' OnLead_MP', 'NotOnLead_MP', 'NS_Geo_MP',
    # 'EW_Geo_MP', 'Declarer_Geo_MP', 'Defender_Geo_MP', 'Declarer_Name',
    # 'Declarer_Score', 'Declarer_ParScore', 'Declarer_Pct',
    # 'Declarer_DD_Tricks', 'Declarer_DD_Score', 'Declarer_DD_Pct',
    # 'Declarer_Tricks_DD_Diff', 'Declarer_Score_DD_Diff',
    # 'Declarer_ParScore_DD_Diff', 'Declarer_Pair', 'Defender_Pair'

    assert set(chart_options)-set(database_column_names) == set(), f"Chart options not in database columns: {set(chart_options)-set(database_column_names)}"
    selected_charts = st.sidebar.multiselect('Select charts to display', chart_options, default=chart_options, key=key_prefix+'-Charts')

    special_columns_unaggregated = {
        "Declarer_DD_GE":(True, "CASE WHEN Tricks >= Declarer_DD_Tricks THEN 1 ELSE 0 END","Declarer_DD_GE"),
        "Declarer_ParScore_GE":(True, "CASE WHEN Declarer_Score >= Declarer_ParScore THEN 1 ELSE 0 END","Declarer_ParScore_GE"),
        "OverTricks":(True, "CASE WHEN Result > 0 THEN 1 ELSE 0 END","OverTricks"),
        "JustMade":(True, "CASE WHEN Result = 0 THEN 1 ELSE 0 END","JustMade"),
        "UnderTricks":(True, "CASE WHEN Result < 0 THEN 1 ELSE 0 END","UnderTricks"),
        }
    
    # todo: looks like there's a bug where columns in mandatory_columns_unaggregated won't show unless they're also in sort_options.
    mandatory_columns_unaggregated = {
        "Date":"Date",
        "Session":"Session",
        "HandRecordBoard":"HandRecordBoard",
        "Declarer_Pair":"Declarer_Pair",
        "Declarer":"Declarer",
        "Declarer_Name":"Declarer_Name",
        "Dummy":"Dummy",
        "Declarer_Score":"Declarer_Score",
        "Declarer_Vul":"Declarer_Vul",
        "Declarer_DD_Score":"Declarer_DD_Score",
        "Declarer_ParScore":"Declarer_ParScore",
        "Declarer_SD_Score":"Declarer_SD_Score",
        "Declarer_SD_Score_Max":"Declarer_SD_Score_Max",
        "Declarer_Pct":"Declarer_Pct",
        "Declarer_DD_GE":special_columns_unaggregated["Declarer_DD_GE"][1],
        "Declarer_ParScore_GE":special_columns_unaggregated["Declarer_ParScore_GE"][1],
        "OverTricks":special_columns_unaggregated["OverTricks"][1],
        "JustMade":special_columns_unaggregated["JustMade"][1],
        "UnderTricks":special_columns_unaggregated["UnderTricks"][1],
        "Declarer_SD_Contract_Max":"Declarer_SD_Contract_Max",
        "Declarer_Tricks_DD_Diff":"Declarer_Tricks_DD_Diff",
        "Declarer_Score_DD_Diff":"Declarer_Score_DD_Diff",
        "Declarer_ParScore_DD_Diff":"Declarer_ParScore_DD_Diff",
        "Declarer_SD_Score_Diff":"Declarer_SD_Score_Diff",
        "Declarer_SD_Score_Max_Diff":"Declarer_SD_Score_Max_Diff",
        "Declarer_ParScore_Pct":"Declarer_ParScore_Pct",
        "Declarer_SD_Pct":"Declarer_SD_Pct",
        "Declarer_SD_Pct_Max":"Declarer_SD_Pct_Max",
        "Declarer_SD_Pct_Diff":"Declarer_SD_Pct_Diff",
        "Declarer_SD_Pct_Max_Diff":"Declarer_SD_Pct_Max_Diff",
        "Declarer_SD_ParScore_Pct_Diff":"Declarer_SD_ParScore_Pct_Diff",
        "Declarer_SD_ParScore_Pct_Max_Diff":"Declarer_SD_ParScore_Pct_Max_Diff",
        #"Table":"'Table'", # todo: Table is reserved word in SQL. What to do?
        #"LoTT":"LoTT",
        }
    if 'club' == club_or_tournament:
        mandatory_columns_unaggregated["Club"] = 'Club'

    # There's many unused columns. See above lists.
    # todo: implement MatchP, Lead, 'Pct', 'Table', 'Score' vs 'Score_NS', 'Round', 'MP_NS', 'MP_EW', LoTT?
    board_scoring_columns = ['Defender_Pair', 'Board', 'Result', 'BidLvl', 'BidSuit', 'Dbl', 'Declarer_Direction', 'Vul','Tricks', 'ContractType', 'Par_Score']
    directional_columns = ['NNum', 'ENum', 'SNum', 'WNum']
    positional_columns = ['Declarer','OnLead','Dummy','NotOnLead']
    master_point_columns = [] # ['Declarer_MP', 'MP_N', 'MP_S', 'MP_E', 'MP_W']
    for col in board_scoring_columns+directional_columns+positional_columns+master_point_columns:
        mandatory_columns_unaggregated[col] = col
    # make all chart_options mandatory
    for k,v in special_columns_unaggregated.items():
        mandatory_columns_unaggregated[k] = v[1] if v[0]==v[2] else f"{v[1]} AS {v[2]}"
    show_cbd_default = False
    columns = bridgestatslib.CreateCheckBoxesFromColumns(database_column_names,mandatory_columns_unaggregated,special_columns_unaggregated,show_cbd_default)
    query = create_query(database_name, '', '', 0, columns, clubs, players, pairs, min_declares, stat_column, minimum_mps, maximum_mps, start_date, end_date) # ','.join(groupby), f"COUNT(*) >= {min_declares}"

    with st.spinner(text="Selecting database rows ..."):
        start_time = time.time()
        query = st.text_input('Sql query', value=query,label_visibility='hidden', key=key_prefix+'-query') # either use initial query or let user change query
        any_position = bridgestatslib.duckdb_arrow_to_df(board_results_arrow, query) # quickly returns the pre-refresh selected_df from cache so keep it virgin-ish. Always use '...' not in selected_df to avoid re-modifing.
        if len(players):
            selected_df = any_position.filter(pl.col('Declarer').is_in(players))
        #elif len(pairs)
        #    selected_df = any_position[any_position['Declarer'].isin(pairs)]
        else:
            selected_df = any_position # position not implemented for pairs
        selected_df_len = len(selected_df)
        #selected_df.insert(selected_df.columns.get_loc('Declarer')+1,'Declarer_Name',selected_df['Declarer'].map(acbl_player_d))
        #selected_df.insert(selected_df.columns.get_loc('Dummy')+1,'Dummy_Name',selected_df['Dummy'].map(acbl_player_d))
        if selected_df_len == 0:
            st.warning('No rows selected. Make sure left sidebar filters are correctly set.')
            st.stop()
        end_time = time.time()
        st.info(f"Query completed in {round(end_time-start_time,2)} seconds. Database has {board_results_len} rows. {selected_df_len} rows selected and aggregated.")

    with st.spinner(text="Preparing data columns ..."):
        start_time = time.time()
        if 'Players' not in selected_df:
            selected_df = selected_df.with_columns([
                pl.col('Declarer_Pair').map_elements(lambda x: [acbl_player_d[n] for n in x.split('_')],return_dtype=pl.List(pl.Utf8)).alias('Players')
            ])
        for player,i in [('Player1',0),('Player2',1)]: # column name, column index
            if player not in selected_df:
                selected_df = selected_df.with_columns([
                    pl.col('Players').map_elements(lambda x: x[i],return_dtype=pl.Utf8).alias(player)
                ])
        if 'HandRecordBoard' in selected_df and 'board_record_string' not in selected_df:
            selected_df = selected_df.with_columns([
                #pl.col("HandRecordBoard").replace_strict(hrd).alias('board_record_string')
                pl.col('HandRecordBoard').map_elements(lambda x: hrd[x],return_dtype=pl.Utf8).alias('board_record_string')
            ])
            # todo: looks like 2500 hrd contains 2500 hand records with superceded hand record ids. Dropping dups here, keeping latest. But this step should be done in hand_record_clean. 
            selected_df = (selected_df
                .sort(['board_record_string', 'Declarer', 'HandRecordBoard'])
                .unique(
                    subset=['board_record_string', 'Declarer'],
                    maintain_order=True,
                    keep='last'
                )
            )
        if 'Count' not in selected_df.columns:
            selected_df = selected_df.with_columns([
                pl.lit(0).alias('Count')
            ])
        # selected_df.drop([k for k,v in cb_d.items() if k in selected_df and not cb_d[k]],axis='columns',inplace=True)
        selected_df = selected_df.select([
            col for col in selected_df.columns if not col.startswith('__')
        ])

        if groupby[0] == 'Session':
            grouped = selected_df.group_by('Session')
        else:
            grouped = selected_df.group_by('Declarer')
        end_time = time.time()
        st.info(f"Data columns completed in {round(end_time-start_time,2)} seconds. Database has {board_results_len} rows. {selected_df_len} rows selected and aggregated.")

    table, chart = st.tabs(['Data Tables', 'Charts'])

    with table:

        with st.spinner(text="Creating data table ..."):

            start_time = time.time()

            # todo: do this when no players (all)?
            # todo: create 2nd table by pairs
            if len(players):
                d = {}
                cols = ['Player','Player_Name','Count','N','E','S','W','N_Pct','E_Pct','S_Pct','W_Pct','Declarer','OnLead','Dummy','NotOnLead','Declarer_Pct','OnLead_Pct','Dummy_Pct','NotOnLead_Pct']
                for col in cols:
                    d[col] = []
                for player in players:
                    d['Player'].append(player)
                    player_df = any_position.filter(pl.col('Declarer').eq(player))
                    d['Player_Name'].append(player_df.select('Declarer_Name').tail(1).row(0)[0])
                    pos_total = 0
                    for pos in ['Declarer','OnLead','Dummy','NotOnLead']:
                        pos_sum = any_position[pos].eq(player).sum()
                        d[pos].append(pos_sum)
                        pos_total += pos_sum
                    for pos in ['Declarer','OnLead','Dummy','NotOnLead']:
                        d[pos+'_Pct'].append(d[pos][-1]/pos_total)
                    direction_total = 0
                    for direction in ['NNum','ENum','SNum','WNum']:
                        direction_sum = any_position[direction].eq(player).sum()
                        d[direction[0]].append(direction_sum)
                        direction_total += direction_sum
                    for direction in ['NNum','ENum','SNum','WNum']:
                        d[direction[0]+'_Pct'].append(d[direction[0]][-1]/direction_total)
                    assert pos_total == direction_total
                    d['Count'].append(pos_total)
                st.info(f"Frequency of Player Positions")
                df = pl.DataFrame(d)
                streamlitlib.ShowDataFrameTable(df,round=2)
                del df


            if len(players) == 0 and len(pairs) == 0:
                #df = grouped.agg({'Date':'first','Declarer_Pair':'first','Count':'count','Player1':'last','Player2':'last','Players':'last'}|{col:'mean' for col in sort_options}).reset_index()
                df = grouped.agg([
                    pl.col('Date').first().alias('Date'),
                    pl.col('Declarer').first().alias('Declarer'),
                    pl.col('Declarer_Name').first().alias('Declarer_Name'),
                    pl.col('Count').count().alias('Count'),
                    pl.col('Player1').last().alias('Player1'), # take a player's last used name
                    pl.col('Player2').last().alias('Player2'), # take a player's last used name
                    *[pl.col(col).mean().alias(col) for col in sort_options]
                ])
                table_df = df.filter(pl.col("Count") >= min_declares).sort(sort_column).head(top_ranked)
                st.info(f"Table of {selected_df_len} rows sorted by {sort_column}. Top performing {len(table_df)} {pair_or_player}s shown.")
                streamlitlib.ShowDataFrameTable(table_df.sort(sort_column, descending=True), color_column=sort_column, round=2)
                del df, table_df

            else:

                # show table for each declarer of each {pair_or_player}
                # todo: for pairs, output tables in Declarer_Pair order. Not Declarer_Name order.
                declarers = selected_df.select('Declarer').unique(maintain_order=True)
                for declarer in declarers.to_series():
                    player_df = selected_df.filter(pl.col('Declarer').eq(declarer))
                    st.info(f"Boards played by {player_df.select('Declarer_Name').tail(1).row(0)}. Sorted by {sort_column}. {player_df.height} boards found.")
                    streamlitlib.ShowDataFrameTable(player_df.sort(sort_column, descending=True), color_column=sort_column, round=2)

                table_df = selected_df.group_by('Session').agg([
                        pl.col('Declarer_Pair').last().alias('Declarer_Pair'),
                        pl.col('Declarer').last().alias('Declarer'),
                        pl.col('Declarer_Name').last().alias('Declarer_Name'),
                        pl.col('Count').count().alias('Count'),
                        *[pl.col(col).mean().alias(col) for col in sort_options]
                    ])
                st.info(f"Means of boards played by {pair_or_player}s aggregated per session. Sorted by {sort_column}.")
                streamlitlib.ShowDataFrameTable(table_df.sort(sort_column, descending=True), color_column=sort_column, round=2)
                del table_df, declarers, player_df

                # head to head analysis
                if len(players) > 1 or len(pairs) > 1:

                    # 1. Count rows per group.
                    group_counts = selected_df.group_by(["Date", "Session", "HandRecordBoard"]).agg(
                        pl.count().alias("group_count")
                    )

                    # 2. Join the counts back to the original DataFrame.
                    table_df = selected_df.join(group_counts, on=["Date", "Session", "HandRecordBoard"])

                    # 3. Filter to keep only groups with more than 1 row.
                    table_df = table_df.filter(pl.col("group_count") > 1)

                    # 4. Optionally, drop the helper column.
                    table_df = table_df.drop("group_count")
                    
                    # 5. Create group key and ngroup
                    table_df = table_df.with_columns(
                        pl.concat_str(["Date", "Session", "HandRecordBoard"], separator="_").alias("group_key")
                    )
                    group_keys = table_df.select("group_key").sort("group_key").unique(maintain_order=True).with_row_count("ngroup")
                    table_df = table_df.join(group_keys, on="group_key").drop("group_key")

                    # 6. First sort within each group by Declarer_Name
                    table_df = table_df.sort(["Declarer_Name"]).group_by("ngroup").agg([
                        pl.all().sort_by("Declarer_Name")
                    ]).explode(pl.all().exclude("ngroup"))

                    # 7. Then sort the entire DataFrame by ngroup and move ngroup to last column
                    table_df = (table_df
                                .sort("ngroup")
                                .select([col for col in table_df.columns if col != "ngroup"] + ["ngroup"]))

                    n_boards = table_df.select(pl.col("ngroup").max()).item() + 1
                    n_sessions = table_df.select(pl.col("Session")).n_unique()

                    st.info(
                        f"Comparison of results of identical boards played by {pair_or_player}s. "
                        f"{n_boards} boards found in {n_sessions} sessions. Sorted by Date, Session, HandRecordBoard, Declarer_Name."
                    )

                    streamlitlib.ShowDataFrameTable(table_df, color_column=sort_column, ngroup_name="ngroup", round=2, key=f"polars_identical_boards_table")

                    # 7. Aggregate per session (grouping by 'Declarer')
                    st.info(
                        f"Comparison of results of identical boards played by {pair_or_player}s aggregated per session. "
                        f"{n_boards} boards found in {n_sessions} sessions. Sorted by {sort_column}."
                    )

                    # 8. Build aggregation expressions:
                    agg_exprs = [
                        pl.col("Declarer_Pair").last().alias("Declarer_Pair"),
                        #pl.col("Declarer").last().alias("Declarer"),
                        pl.col("Declarer_Name").last().alias("Declarer_Name"),
                        pl.col("Count").count().alias("Count"),
                    ] + [pl.col(col).mean().alias(col) for col in sort_options]

                    table_df_grouped = table_df.group_by("Declarer").agg(agg_exprs)

                    # 9. Sort the aggregated result.
                    if sort_column in table_df_grouped.columns:
                        table_df_grouped = table_df_grouped.sort(sort_column, descending=True)
                    else:
                        table_df_grouped = table_df_grouped.sort(["Player1", "Player2"])

                    streamlitlib.ShowDataFrameTable(table_df_grouped, color_column=sort_column, round=2, key=f"polars_aggregated_per_session_table")

                    del table_df, table_df_grouped, group_counts, group_keys

                    # 10. Head-to-head comparison for pairs.
                    if pair_or_player == 'pair':

                        # Get unique columns to avoid duplicates in join
                        unique_columns = ["Date", "Session", "HandRecordBoard", "Declarer_Pair"]
                        sort_columns = unique_columns + ["Declarer_Name"]

                        # Step 1: Remove duplicates based on the columns of interest,
                        #         maintaining the order determined by sort_columns.
                        unique_df = selected_df.sort(sort_columns).unique(unique_columns, maintain_order=True)

                        # Step 2: Compute group counts per ["Date", "Session", "HandRecordBoard"].
                        group_counts = unique_df.group_by(["Date", "Session", "HandRecordBoard"]).agg(
                            pl.count().alias("group_count"),
                            pl.col("Declarer").alias("Declarers")
                        )

                        # Step 3: Filter out groups with only one row.
                        valid_groups = group_counts.filter(pl.col("group_count") > 1).select(["Date", "Session", "HandRecordBoard", 'Declarers', 'group_count'])

                        # Step 4: Keep only rows from groups with more than one row.
                        filtered_df = unique_df.join(valid_groups, on=["Date", "Session", "HandRecordBoard"], how="inner")

                        # Step 5: Self-join to pair up all declarers
                        h2h_df = (
                            filtered_df.join(
                                filtered_df.select(["HandRecordBoard", "Declarer", "Declarer_Name"] + sort_options),
                                on="HandRecordBoard",
                                suffix="_compare"
                            )
                            # Filter out self-matches and keep only unique pairs
                            .filter(pl.col("Declarer") != pl.col("Declarer_compare"))
                            .unique(subset=["HandRecordBoard", "Declarer", "Declarer_compare"], maintain_order=True)
                        )

                        # Step 6: Create columns for the head-to-head key
                        h2h_df = h2h_df.with_columns([
                            (pl.col("Declarer") + "_" + pl.col("Declarer_compare")).alias("H2H"),
                            (
                                pl.when(pl.col("Declarer") < pl.col("Declarer_compare"))
                                .then(pl.col("Declarer") + "_" + pl.col("Declarer_compare"))
                                .otherwise(pl.col("Declarer_compare") + "_" + pl.col("Declarer"))
                        ).alias("H2H_sorted")
                        ])

                        # Step 7: Group by declarer pairs and aggregate
                        h2h_df = h2h_df.group_by(["H2H", "H2H_sorted"]).agg([
                            pl.col("HandRecordBoard").count().alias("Count"),
                            pl.col("Declarer").first().alias("Declarer1"),
                            pl.col("Declarer_compare").first().alias("Declarer2"),
                            pl.col("Declarer_Name").first().alias("Declarer_Name"),
                            pl.col("Declarer_Name_compare").first().alias("Declarer_Name2"),
                            *[pl.col(col).mean().alias(col) for col in sort_options]
                        ])

                        # Step 8: Sort and create ngroup
                        h2h_df = (
                            h2h_df
                            # Create a list of names for each group
                            .with_columns([
                                pl.concat_list([pl.col("Declarer_Name"), pl.col("Declarer_Name2")]).alias("names")
                            ])
                            # Sort names within each list and join them
                            .with_columns([
                                pl.col("names").map_elements(lambda x: "_".join(sorted(x))).alias("group_key")
                            ])
                            # Sort by the group key and Declarer_Name
                            .sort(["group_key", "Declarer_Name"])
                            # Create ngroup based on group key
                            .with_columns([
                                pl.col("group_key").rank("dense").alias("ngroup")
                            ])
                            .drop(["names", "group_key"])
                            .select([
                                col for col in h2h_df.columns 
                                if col not in ["ngroup", "H2H_sorted", "Declarer1", "Declarer2", "Declarer_Name2"]
                            ] + ["Declarer1", "Declarer2", "Declarer_Name2", "H2H_sorted", "ngroup"])
                        )

                        st.info(
                            "Comparison of head-to-head results of identical boards played between pairs aggregated per boards. Sorted by Declarer_Name."
                        )
                        streamlitlib.ShowDataFrameTable(h2h_df, color_column=sort_column, ngroup_name="ngroup", round=2)
                        del h2h_df, valid_groups, filtered_df, unique_df, group_counts

            end_time = time.time()
            st.info(f"Data table created in {round(end_time-start_time,2)} seconds.")

    with chart:

        with st.spinner(text="Creating charts ..."):

            start_time = time.time()

            # using st.write because it displays in a suitable font size and style.
            st.write(f"Acronyms: BidLvl is Contract Level, BidSuit is Contract Suit, ContractType is Type of Contract (passed-out, partial, game, small slam, grand slam), Dbl is Doubled, MP is Player's Master Points Pct is Match Point Percent")

            # removed feature because of annoying issue with new query of using groupby list but no special_columns translation.
            # declarer_groups = selected_df.group_by(groupby).groups
            # if len(declarer_groups) <= 10: # if 10 or fewer, revert to unaggregated.
                # Non-aggregated
                # query = create_query(database_name, '', '', 0, ','.join(groupby+selected_charts), clubs, players, pairs, min_declares, stat_column, minimum_mps, maximum_mps, start_date, end_date, 'acbl_player_df')
                # selected_df = duckdb.arrow(board_results_arrow).query('board_results', query).to_df()  # todo: can this be cached?
            # else:
            #    selected_df = selected_df.sample(1000)
            bridgestatslib.ShowCharts(selected_df,selected_charts,stat_column)

            # Get column names that match the pattern
            pct_columns = selected_df.select(pl.col(r'^.*(Pct|Pct_Max)$')).columns

            # Use the column names in ShowCharts
            bridgestatslib.ShowCharts(
                selected_df.select(pl.col(r'^.*(Pct|Pct_Max)$')),
                [','.join(pct_columns)]  # Join the column names, not the Series
            )

            # Get column names that match the pattern
            diff_columns = selected_df.select(pl.col(r'^.*Pct.*Diff.*$')).columns

            # Use the column names in ShowCharts
            bridgestatslib.ShowCharts(
                selected_df.select(pl.col(r'^.*Pct.*Diff.*$')),
                [','.join(diff_columns)]  # Join the column names, not the Series
            )

            end_time = time.time()
            st.info(f"Charts created in {round(end_time-start_time,2)} seconds.")


