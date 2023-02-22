
# todo:

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


@st.cache_resource()
def load_club_hand_records(filename):
    hand_records_arrow = pq.read_table(filename)
    return hand_records_arrow


@st.cache_resource()
def load_club_board_results(filename):
    arrow_table = pq.read_table(filename)
    return arrow_table


@st.cache_resource()
def load_club_player_d(filename):
    return pd.read_pickle(filename)


@st.cache_resource()
def load_club_hand_records_d(filename):
    return pd.read_pickle(filename)


@st.cache_resource()
def load_tournament_hand_records(filename):
    return pq.read_table(filename)


@st.cache_resource()
def load_tournament_board_results(filename):
    return pq.read_table(filename)


@st.cache_resource()
def load_player_info_df(filename):
    return pd.read_parquet(filename)


@st.cache_resource()
def load_club_df(filename):
    return pd.read_parquet(filename)


@st.cache_resource()
def load_tournament_player_d(filename):
    return pd.read_pickle(filename)


@st.cache_resource()
def load_tournament_hand_records_d(filename):
    return pd.read_pickle(filename)


# uncacheable due to arrow_table. just leave as a helper function.
#@st.cache_resource()
def duckdb_arrow_to_df(_arrow_table, query):
    return duckdb.arrow(_arrow_table).query('arrow_table', query).to_df()


# todo: is this obsolete? Seems to freeze or slow webpages.
#@st.cache_resource()
def duckdb_query(query):
    return duckdb.query(query).to_df()

    
def CreateCheckBoxesFromColumns(column_names,mandatory_columns,special_columns,show_cbd_default):
    cb_d = mandatory_columns
    columns = ', '.join([v if k not in special_columns else special_columns[k][1] if special_columns[k][2] is None else special_columns[k][1]+' AS '+special_columns[k][2] for k,v in cb_d.items()]) # required columns
    for col in list(column_names)+list(special_columns.keys()):
        as_name = col
        if col not in cb_d and not col.startswith('__'):
            if col in special_columns:
                show_cb = special_columns[col][0]
                value = special_columns[col][1]
                if special_columns[col][2] is not None:
                    as_name = special_columns[col][2]
                    value += ' AS '+as_name
            else:
                show_cb = show_cbd_default
                value = col
            if show_cb:
                checked = st.sidebar.checkbox(as_name,value=show_cb,key='Player-CheckBox-'+as_name)
                cb_d[col] = checked
                if checked:
                    if columns == '':
                        columns = value
                    else:
                        columns += ', '+value
    return columns


def ShowCharts(selected_df,selected_charts,stat_column=None):

    available_charts = []
    for chart in selected_charts:
        stat_column_split = chart.replace(' ','').split(',')
        if all([col in selected_df for col in stat_column_split]):
            available_charts.append(stat_column_split)

    selected_df_len = len(selected_df)
    if 'Declarer' in selected_df:
        declarer_groups = selected_df.groupby(['Declarer','Declarer_Name']).groups
        st.info(f"Selected: Unique declarers:{len(declarer_groups)} rows:{selected_df_len} charts:{available_charts}")
    else:
        declarer_groups = {}

    figsize = (26,2)
    if 0 < len(declarer_groups) <= 10: # if 10 or fewer declarers, revert to unaggregated.

        # create charts with x-axis having a column for each player
        for stat_column_split in available_charts:
            if len(stat_column_split) == 1: # todo: implement charts having more than one x axis column. e.g. HCP_.*
                players_d = {}
                for col in stat_column_split:
                     for k,v in declarer_groups.items():
                        s = selected_df.loc[v][col]
                        vc = s.value_counts(normalize=True).sort_index() # initialize here to suppress ide warning
                        if pd.api.types.is_float_dtype(s):
                            for r in [2,1,0,-1]:
                                vc = s.astype('float64').round(r).value_counts(normalize=True).sort_index()
                                if len(vc) <= 100: # try more rounding if too crowded.
                                    break
                        players_d['('+','.join([str(k[0]),k[1],str(len(vc))])+')'] = vc # caution: tricky to form str.
                title = f"Frequency Percentage of {', '.join(stat_column_split)} {', '.join(players_d.keys())}"
                ax = pd.DataFrame(players_d).plot(kind='bar',figsize=figsize,title=title)
                ax.legend(title='(Player Number, Player Name, Rows Found, Mean)')
                st.pyplot(plt,clear_figure=True)
            
    else:
    
         for stat_column_split in available_charts:
                                    
            if len(stat_column_split) == 3: # 3 variable heat map
                cross_table = selected_df.groupby([stat_column_split[0],stat_column_split[1]])[stat_column_split[2]].mean().unstack()
                streamlitlib.plot_heatmap(cross_table, zlabel=stat_column_split[2])
                del cross_table

            else:

                d = {}
                for col in stat_column_split:
                    if pd.api.types.is_float_dtype(selected_df[col]):
                        # todo: this is a kludge to limit the number of x axis labels. Attempting to maximize performance and precision and minimize overly crowded x axis.
                        for r in [2,1,0,-1]:
                            d[col] = selected_df[col].astype('float64').round(r).value_counts(normalize=True).sort_index()
                            if len(d[col]) <= 100: # try more rounding if too crowded.
                                break
                    else:
                        d[col] = selected_df[col].value_counts(normalize=True).sort_index()
                ax = pd.DataFrame(d).plot(kind='bar',xlabel=stat_column_split,ylabel=f"Percentage Frequency",title=f"Frequency of {', '.join(d.keys())} values. {selected_df_len} observations.",figsize=figsize)
                st.pyplot(plt,clear_figure=True)
                del d
                
                if False: # altair charting works but needs some niceification e.g. width.
                    chart_df = selected_df[stat_column_split].melt(var_name='column')
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x='column',
                        y='count()',
                        column='value:O',
                        color='column'
                    )
                    st.altair_chart(chart)
                    del split_df, chart_df
