
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
import streamlitlib # must be placed after sys.path.append. vscode re-format likes to move this to the top


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
    st.caption(f"Data read completed in {round(end_time-start_time,2)} seconds.  {len(acbl_player_df)} rows read.")

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

selected_df = acbl_player_df.convert_dtypes()
selected_df['mp_total'] = selected_df['mp_total'].astype('float64') # seems like the only column which convert_dtypes fails to please
#print(selected_df.info())
selected_df = selected_df if len(clubs_regex)==0 else selected_df[selected_df['club'].astype('string').str.contains(clubs_regex,regex=True)]
selected_df = selected_df if len(player_numbers_regex)==0 else selected_df[selected_df['acbl_number'].astype('string').str.contains(player_numbers_regex,regex=True)]
selected_df = selected_df if len(player_names_regex)==0 else selected_df[selected_df['last_name'].str.contains(player_names_regex,case=False,regex=True)]
selected_df = selected_df.drop(selected_df.filter(regex=r'mp_').columns,axis='columns') # drop master point columns per ACBL privacy requirements

table, charts = st.tabs(["Data Table", "Charts"])

st.caption(f"Database has {len(acbl_player_df)} rows. {len(selected_df)} rows selected.")

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
        
    with st.spinner(text="Creating charts ..."):
        start_time = time.time()
        # looks best. probably worth investing time. highly configurable. can eliminate need for sort_value, value_counts, aggregate, ...
        # apparently altair is an api for python while vega is for javascript.
        # https://altair-viz.github.io/user_guide/API.html
        #st.caption("Using altair charting - ")
        # count() eliminates need for value_counts. sort='-x' eliminates need for sort_values(ascending=False).
        #for col in selected_df.columns:

        col = 'rank_description'
        x = f'count({col}):Q'
        y = f'{col}:N'
        title_x = 'Count'
        title_y = col.replace('_',' ').title()
        title = f'Frequency of {title_y}'
        sort_y = '-x'
        c = ( # use () to enclose break up a dotted expression into multiple lines
            alt.Chart(selected_df)
            .mark_bar()
            .encode(alt.X(x, title=title_x),alt.Y(y, title=title_y, sort=sort_y)) # configure x,y data #  sort1=sort
            .properties(width=1000, height=500,title=title) # provide chart title. Haven't found autosizing to work. 
            .configure_axis(labelFontSize=14, titleFontSize=20, labelLimit=200, titleFontWeight='bold') # configure axis titles and labels
            .configure_title(fontSize=20, offset=5, orient='top', anchor='middle') # configure chart title
            )
        #alt.AutoSizeParams(type='fit-y') # doesn't work
        # specifying use_container_width=True causes double rendering. Avoid by specifying .properties(width=1000, height=500, ...
        st.altair_chart(c) # use_container_width=True)

        if False: # experiments with other charting software
            # pyplot not as pretty as altair
            st.caption("Using pyplot charting")
            ax = selected_df.value_counts('rank_description').sort_values().plot(kind='barh',title=f"Rank Achievments of {len(selected_df)} players")
            st.pyplot(plt,clear_figure=True)

            # Simple to use but very limited - no horizontal bar feature
            st.caption("Using st.bar_charting - Doesn't do horizontal")
            st.bar_chart(selected_df.value_counts('rank_description').sort_values())

            # attempt to swap x,y but doesn't display correctly.
            st.caption("st_bar_chart horizontal - fail")
            chart_df = selected_df.value_counts('rank_description')
            chart_df = pd.Series(chart_df.index, index=chart_df).sort_index(ascending=False)
            #chart_df = selected_df.pivot(index="rank_description", columns='Direction') #, values=['HasFrontAndRearDetect','HasOnlyFrontDetect','HasOnlyRearDetect']).reset_index()
            #print(chart_df)
            st.bar_chart(chart_df) #pd.Series(chart_df, index=chart_df))
                
        end_time = time.time()
        st.caption(f"Charts created in {round(end_time-start_time,2)} seconds.")
