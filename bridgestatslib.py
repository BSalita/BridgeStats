# todo:

import streamlit as st
import pathlib
import pickle
import pyarrow.parquet as pq
import duckdb
import polars as pl
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import time
import sys
sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
import streamlitlib # must be placed after sys.path.append. vscode re-format likes to move this to the top


@st.cache_resource()
def load_club_hand_records(filename):
    return pl.scan_parquet(filename)


@st.cache_resource()
def load_club_board_results(filename):
    return pl.scan_parquet(filename)


@st.cache_resource()
def load_club_player_d(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


@st.cache_resource()
def load_club_hand_records_d(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


@st.cache_resource()
def load_tournament_hand_records(filename):
    return pl.scan_parquet(filename)


@st.cache_resource()
def load_tournament_board_results(filename):
    return pl.scan_parquet(filename)


@st.cache_resource()
def load_player_info_df(filename):
    return pl.scan_parquet(filename)


@st.cache_resource()
def load_club_df(filename):
    return pl.scan_parquet(filename)


@st.cache_resource()
def load_tournament_player_d(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


@st.cache_resource()
def load_tournament_hand_records_d(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Helper function for DuckDB queries
def duckdb_arrow_to_df(board_results_arrow, query):
    # Convert LazyFrame to Arrow table first
    arrow_table = board_results_arrow.collect().to_arrow()
    return pl.from_arrow(duckdb.arrow(arrow_table).query('arrow_table', query).arrow())


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


# todo: remove stat_column in favor of column_filter?
def ShowCharts(selected_df,selected_charts,stat_column=None,column_filter='.*'):
    # Convert polars DataFrame to pandas for plotting
    # This is temporary until we implement native polars plotting
    selected_df_pd = selected_df.to_pandas()
    
    available_charts = []
    for chart in selected_charts:
        stat_column_split = chart.replace(' ','').split(',')
        if all([col in selected_df.columns for col in stat_column_split]):
            available_charts.append(stat_column_split)

    selected_df_len = len(selected_df)
    if 'Declarer' in selected_df.columns:
        declarer_groups = (
            selected_df
            .group_by(['Declarer', 'Declarer_Name'])
            .agg(pl.count())
        )
        st.info(f"Selected: Unique declarers:{len(declarer_groups)} rows:{selected_df_len} charts:{available_charts}")
    else:
        declarer_groups = pl.DataFrame()

    figsize = (26,2)
    if 0 < len(declarer_groups) <= 10:
        # For small number of declarers, show unaggregated data
        for stat_column_split in available_charts:
            if len(stat_column_split) == 1:
                players_d = {}
                for col in stat_column_split:
                    for row in declarer_groups.iter_rows(named=True):
                        declarer, declarer_name = row['Declarer'], row['Declarer_Name']
                        s = selected_df.filter(pl.col('Declarer') == declarer)[col]
                        
                        # Convert to pandas for value_counts operation
                        s_pd = s.to_pandas()
                        if pl.Series(s).dtype in [pl.Float32, pl.Float64]:
                            for r in [2,1,0,-1]:
                                vc = s_pd.round(r).value_counts(normalize=True).sort_index()
                                if len(vc) <= 100:
                                    break
                        else:
                            vc = s_pd.value_counts(normalize=True).sort_index()
                        players_d[f'({declarer},{declarer_name},{len(vc)})'] = vc
                
                title = f"Frequency Percentage of {', '.join(stat_column_split)} {', '.join(players_d.keys())}"
                ax = pd.DataFrame(players_d).plot(kind='bar',figsize=figsize,title=title)
                ax.legend(title='(Player Number, Player Name, Rows Found, Mean)')
            else:
                selected_cols = selected_df.select(stat_column_split).to_pandas()
                ax = selected_cols.hist(figsize=figsize)
            st.pyplot(plt,clear_figure=True)
    else:
        for stat_column_split in available_charts:
            if len(stat_column_split) == 3:
                # 3 variable heat map
                cross_table = (
                    selected_df
                    .group_by([stat_column_split[0], stat_column_split[1]])
                    .agg(pl.col(stat_column_split[2]).mean())
                    .collect()
                    .pivot(
                        values=stat_column_split[2],
                        index=stat_column_split[0],
                        columns=stat_column_split[1]
                    )
                    .to_pandas()
                )
                streamlitlib.plot_heatmap(cross_table, zlabel=stat_column_split[2])
                del cross_table
            else:
                d = {}
                for col in stat_column_split:
                    s = selected_df.select(col).to_pandas()[col]
                    if pd.api.types.is_float_dtype(s):
                        for r in [2,1,0,-1]:
                            d[col] = s.round(r).value_counts(normalize=True).sort_index()
                            if len(d[col]) <= 100:
                                break
                    else:
                        d[col] = s.value_counts(normalize=True).sort_index()
                
                ax = pd.DataFrame(d).plot(
                    kind='bar',
                    xlabel=stat_column_split,
                    ylabel="Percentage Frequency",
                    title=f"Frequency of {', '.join(d.keys())} values. {selected_df_len} observations.",
                    figsize=figsize
                )
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
