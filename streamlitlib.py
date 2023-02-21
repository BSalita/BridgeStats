import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode, AgGridTheme, JsCode


def widen_scrollbars():
    st.markdown("""
                    <html>
                        <head>
                        <style>
                            ::-webkit-scrollbar {
                                width: 14px;
                                height: 14px;
                                }

                                /* Track */
                                ::-webkit-scrollbar-track {
                                background: #f1f1f1;
                                }

                                /* Handle */
                                ::-webkit-scrollbar-thumb {
                                background: #888;
                                }

                                /* Handle on hover */
                                ::-webkit-scrollbar-thumb:hover {
                                background: #555;
                                }
                        </style>
                        </head>
                        <body>
                        </body>
                    </html>
                """, unsafe_allow_html=True)
                
                
def style_table():

    rows = {
        "selector":"tbody tr:nth-child(even)",
        'props': 'background-color: lightgrey; color: black;'
        }

    odd_hover = {  # for row hover use <tr> instead of element <td>
        'selector': 'tr:nth-child(odd):hover',
        'props': [('background-color', '#ffffb3'),('cursor','pointer')]
    }

    even_hover = {  # for row hover use <tr> instead of element <td>
        'selector': 'tr:nth-child(even):hover',
        'props': [('background-color', '#ffd700'),('cursor','pointer')]
    }

    index_names = {
        'selector': '.index_name',
        'props': [('background-color', '#ffffb3')]
    }
    
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: lightblue; color: black;'
    }

    return [rows, odd_hover, even_hover, index_names, headers]
    
    
def plot_heatmap(cross_table, fmt='.2f', xlabel=None, ylabel=None, zlabel=None, title=None):
    if xlabel is None: xlabel = cross_table.columns.name
    if ylabel is None: ylabel = cross_table.index.name
    if title is None: title = f'Influence of {xlabel} and {ylabel} on {zlabel}'
    fig, ax = plt.subplots(figsize=(8,8))

    #sns.set(font_scale=2) # font size 2
    sns.heatmap(cross_table,
                annot=True,
                annot_kws={"size": 4},
                fmt=fmt,
                cmap='rocket_r',
                linewidths=.5,
                ax=ax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    st.pyplot(plt,clear_figure=True)


def ShowDataFrameTable(table_df,output_method='aggrid',ngroup_name=None,round=2):

    if output_method == 'table':
        st.table(table_df.style.format({col:'{:,.2f}' for col in table_df.select_dtypes('float')}).set_table_styles(style_table())) #,1600,500)

    elif output_method == 'dataframe':
        #st.dataframe(table_df) #,1600,1000) # arbitrary 1600x1000 pixels
        st.dataframe(table_df.style.format({col:'{:,.2f}' for col in table_df.select_dtypes('float')}).set_table_styles(style_table()),1600,500)

    elif output_method == 'aggrid': # todo: current code doesn't adjust for dark mode
        # https://github.com/PablocFonseca/streamlit-aggrid/blob/main/st_aggrid/__init__.py#L190
 
        gb = GridOptionsBuilder.from_dataframe(table_df)
        #gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
        #gb.configure_side_bar() #Add a sidebar
        #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
        gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True, wrapHeaderText=True, autoHeaderHeight=True)
        custom_css = {".ag-header-cell-text": {"font-size": "12px", 'text-overflow': 'revert;', 'font-weight': 700},
          ".ag-theme-streamlit": {'transform': "scale(0.8)", "transform-origin": '0 0'}}
        gridOptions = gb.build()
        # ngroup_name is the name of a column which contains the same value for every members of a group. Used to alternate colors.
        if ngroup_name is not None:
            jscode = """
                function(params) {
                    if (params.data.ngroup_name%2 === 0) {
                        return {
                            'color': 'white',
                            'backgroundColor': 'AntiqueWhite'
                        }
                    }
                };
            """.replace('ngroup_name',ngroup_name)
            gridOptions['getRowStyle'] = JsCode(jscode)
        if round: # a bit dangerous as it introduces a hidden side-effect of modifying the dataframe by rounding.
            for col in table_df.select_dtypes('float'): # rounding or {:,.2f} only works on float64!
                table_df[col] = table_df[col].astype('float64').round(round)
        AgGrid(
            table_df,
            gridOptions=gridOptions,
            custom_css=custom_css,
            allow_unsafe_jscode=True, # needed for jscode
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            #data_return_mode='AS_INPUT', 
            #update_mode='MODEL_CHANGED', 
            #fit_columns_on_grid_load=True, # deprecated?
            theme=AgGridTheme.BALHAM, # Only choices: AgGridTheme.STREAMLIT, AgGridTheme.ALPINE, AgGridTheme.BALHAM, AgGridTheme.MATERIAL
            #enable_enterprise_modules=True,
            height=330 if len(table_df) > 10 else 50+len(table_df)*30 # not sure why 50 is right height but scoll bars disappear using both 50/*30.
            #width='100%',
            #reload_data=True
            )
            
    else:
        st.error(f"ShowDataFrameTable: Unknown output method: {output_method}")

