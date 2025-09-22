import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # or DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def print_to_log_info(*args):
    print_to_log(logging.INFO, *args)
def print_to_log_debug(*args):
    print_to_log(logging.DEBUG, *args)
def print_to_log(level, *args):
    logging.log(level, ' '.join(str(arg) for arg in args))

import streamlit as st
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode, AgGridTheme, JsCode
import threading
import time
import queue


@st.cache_resource
def safe_resource():
    return {
        'lock': threading.Lock(),
        'waiting_count': 0,
        'queue': queue.Queue()  # FIFO queue
    }

# this version of perform_queued_work() uses self for class compatibility, older versions did not.
def perform_queued_work(self, work_fn, work_description):
    """
    Perform work in a FIFO queue with progress tracking.
    
    Args:
        work_fn: Callable that takes df and **work_kwargs as arguments
        work_description: Description of work for status messages
    """
    # Create empty placeholders
    status_placeholder = st.empty()
    queue_placeholder = st.empty()

    acquired = False
    resources = safe_resource()
    mutex = resources['lock']
    wait_queue = resources['queue']
    my_token = object()  # Unique token for this session

    try:
        # Add ourselves to the queue and increment counter
        with threading.Lock():
            wait_queue.put(my_token)
            resources['waiting_count'] += 1
            
        while not acquired:
            # Only try to acquire if we're at the front of the queue
            if wait_queue.queue[0] == my_token:
                acquired = mutex.acquire(timeout=2)
                if acquired:
                    wait_queue.get()  # Remove ourselves from queue
            
            if not acquired:
                status_placeholder.info(f"{work_description} is queued for processing. Please wait...")
                position = list(wait_queue.queue).index(my_token) + 1
                queue_placeholder.info(f"Your position in queue: {position} of {resources['waiting_count']}")
                time.sleep(1)

        # Decrement waiting count when acquired
        with threading.Lock():
            resources['waiting_count'] -= 1
            
        status_placeholder.empty()
        queue_placeholder.empty()
        progress = st.progress(0)
        
        # Call the work function with progress bar and other arguments
        df = work_fn()

    finally:
        if acquired:
            mutex.release()
        # Clean up queue if we're still in it
        with threading.Lock():
            if my_token in wait_queue.queue:
                # Remove our token (this is a bit hacky but necessary for cleanup)
                items = list(wait_queue.queue)
                items.remove(my_token)
                wait_queue.queue.clear()
                for item in items:
                    wait_queue.put(item)
                resources['waiting_count'] -= 1

    return df


def move_focus():
    # inspect the html to determine which control to specify to receive focus (e.g. text or textarea).
    st.components.v1.html(
        f"""
            <script>
                var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
                for (var i = 0; i < textarea.length; ++i) {{
                    textarea[i].focus();
                }}
            </script>
        """,
    )


def stick_it_good():

    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 0px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True
    )


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


# problems:
# streamlit-aggrid no longer supported.
# AgGrid display always display properly in tab. Solution is not to use it in tabs.
# AgGrid doesn't handle Series.
# st.dataframe is incredibly slow when displaying color styles per cell. Haven't been able to find a way to display alternating row colors, only per cell styling.
# st.table has huge row size.
def ShowDataFrameTable(table_df,key=None,output_method='aggrid',color_column=None,ngroup_name=None,round=2,tooltips=None,height_rows=None):

    # seems like everthing requires a pandas dataframe. table_df.style() doesn't work with polars.
    if isinstance(table_df, pl.DataFrame):
        table_df = table_df.to_pandas()

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
        # some streamlit custom_css examples: https://discuss.streamlit.io/t/how-to-use-custom-css-in-ag-grid-tables/26743
        # patch - 17-Aug-2023 - added - #gridToolBar to fix missing horizontal scrollbar. https://discuss.streamlit.io/t/st-aggrid-horizontal-scroll-bar-disappears-when-i-define-a-height/46217/10?u=bsalita
        # Define column tooltips
        if tooltips is not None:
            for col in table_df.columns:
                gb.configure_column(col, headerTooltip=tooltips.get(col,col))
        custom_css = {
            '.ag-header-cell-text': {'font-size': '12px', 'text-overflow': 'revert;', 'font-weight': 700},
            '.ag-theme-streamlit': {'transform': 'scale(0.8)', 'transform-origin': '0 0'},
            # odd/even not working # '.ag-row-hover(odd)': {'background-color': '#ffffb3', 'cursor': 'pointer'},
            # odd/even not working # '.ag-row-hover(even)': {'background-color': '#ffd700', 'cursor': 'pointer'},
            #'#gridToolBar': {'padding-bottom': '0px !important'}
            }
        if color_column is not None:
            gb.configure_column(color_column, cellStyle={'color': 'black', 'background-color': '#FEFBF7'}) # must be executed before build()
        gridOptions = gb.build()
        # ngroup_name is the name of a column which contains the same value for every member of a group. Used to alternate colors for a group as opposed to odd/even.
        if ngroup_name is None:
            custom_css['.ag-row:nth-child(odd)'] = {'background-color': 'white'}
            custom_css['.ag-row:nth-child(even)'] = {'background-color': 'whitesmoke'}
        else:
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
        # Determine number of rows to size height
        if height_rows is None:
            rows_to_show = min(4, len(table_df))
        else:
            rows_to_show = max(1, min(int(height_rows), len(table_df)))
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
            # height calc is a bit arbitrary. add 50 for header. add 40 per row for text. add 10+2 per row for line separators.  add 20 in case there's a horizontal scrollbar.
            height=50+(30+10+2)*rows_to_show+20,
            #width='100%',
            #reload_data=True
            key=key
            )
            
    else:
        st.error(f"ShowDataFrameTable: Unknown output method: {output_method}")


import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, Flowable, PageBreak
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import markdown
from xml.sax.saxutils import unescape
from bs4 import BeautifulSoup
#import base64

class HorizontalLine(Flowable):
    """A custom flowable that draws a horizontal line."""

    def __init__(self, width):
        Flowable.__init__(self)
        self.width = width

    def draw(self):
        self.canv.line(0, 0, self.width, 0)


def markdown_to_paragraphs(md_string, styles):
    # Convert Markdown to HTML
    html_content = markdown.markdown(md_string)
    
    # Unescape HTML entities
    html_content = unescape(html_content)
    
    # Convert HTML paragraphs to reportlab paragraphs
    paragraphs = []
    for line in html_content.split("\n"):
        if line.startswith("<h"):
            # Extract header level and text
            level = int(line[2])
            text = line[4:-5]
            style = styles[f"Heading{level}"]
        else:
            text = line[3:-4]  # Remove <p> and </p> tags
            style = styles["Normal"]
        
        if text:
            paragraphs.append(Paragraph(text, style))
            paragraphs.append(Spacer(1, 12))
    
    return paragraphs


def dataframe_to_table(df, max_rows: int | None = None, max_cols: int | None = None, shrink_to_fit: bool = False):
    # Convert DataFrame to HTML

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas() # doesn't handle native polars
    # Apply optional slicing
    if max_rows is not None:
        df = df.iloc[:max_rows]
    if max_cols is not None:
        df = df.iloc[:, :max_cols]
    html_content = df.to_html(index=False) # index=False to omit index column
    
    # Parse HTML to extract table data
    soup = BeautifulSoup(html_content, 'html.parser')
    table_data = []
    for row in soup.table.findAll('tr'):
        row_data = []
        for cell in row.findAll(['td', 'th']):
            row_data.append(cell.get_text())
        table_data.append(row_data)
    
    # Create reportlab table with auto-sizing if requested
    if shrink_to_fit and len(table_data) > 0:
        # Calculate available width (landscape letter minus margins)
        from reportlab.lib.pagesizes import landscape, letter
        page_width = landscape(letter)[0]
        available_width = page_width - 72  # 1 inch margins on each side
        
        # Estimate column widths based on content
        col_count = len(table_data[0])
        if col_count > 0:
            # Check if this is a pairs report by looking for 'Pair_Names' column
            headers = table_data[0] if len(table_data) > 0 else []
            pair_names_idx = None
            for i, header in enumerate(headers):
                if 'Pair_Names' in str(header):
                    pair_names_idx = i
                    break
            
            if pair_names_idx is not None:
                # Special column widths for pairs report - make Pair_Names column 2x wider
                base_width = available_width / (col_count + 1)  # Extra space for 2x column
                col_widths = [base_width] * col_count
                col_widths[pair_names_idx] = base_width * 2  # Make Pair_Names 2x wider
                
                # Adjust other columns to fit the remaining space
                remaining_width = available_width - col_widths[pair_names_idx]
                other_width = remaining_width / (col_count - 1)
                for i in range(col_count):
                    if i != pair_names_idx:
                        col_widths[i] = other_width
            else:
                # Default equal column widths
                col_widths = [available_width / col_count] * col_count
            
            table = Table(table_data, repeatRows=1, colWidths=col_widths)
        else:
            table = Table(table_data, repeatRows=1)
    else:
        table = Table(table_data, repeatRows=1)
    
    # Adjust font size for better fit when shrinking
    font_size = 7 if shrink_to_fit else 9
    
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), '#E5E5E5'),
        ('GRID', (0, 0), (-1, -1), 1, '#D5D5D5'),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Align text to top of cells
    ]

    # Alternate row colors for even and odd rows
    for i, _ in enumerate(table_data[1:], start=1):  # Skip header row
        table_style.append(('BACKGROUND', (0, i), (-1, i), 'white' if i % 2 else 'whitesmoke'))
    
    table_style.append(('BACKGROUND', (2, 1), (2, -1), '#FEFBF7')) # 2nd column should be colored
    
    table.setStyle(TableStyle(table_style))
    
    return table


def create_pdf(pdf_assets, title, output_filename=None, max_rows: int | None = None, max_cols: int | None = 11, rows_per_page: int | list[int] | tuple[int, ...] | None = None, shrink_to_fit: bool = False):
    # Create a BytesIO object to capture the PDF data
    buffer = BytesIO()
    
    # Create a new document with the buffer as the destination
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    doc.title = title

    # Create a list to hold the document's contents
    story = []
    
    # Get a sample style sheet
    styles = getSampleStyleSheet()
    
    def _iter_paginated_frames(frame, limit_rows: int | None, per_page_spec):
        # Convert Polars to pandas if needed
        import pandas as pd  # local import for safety
        if isinstance(frame, pl.DataFrame):
            frame = frame.to_pandas()
        elif not isinstance(frame, pd.DataFrame):
            return []

        total_rows = len(frame) if limit_rows is None else min(limit_rows, len(frame))
        if total_rows <= 0:
            return []

        # Determine first-page and subsequent-page row counts
        if per_page_spec is None:
            # default: old behavior of ~30 rows per chunk
            first_count = 30
            subsequent_count = 30
        elif isinstance(per_page_spec, int):
            first_count = per_page_spec
            subsequent_count = per_page_spec
        else:
            seq = list(per_page_spec)
            first_count = seq[0] if len(seq) > 0 else 30
            subsequent_count = seq[1] if len(seq) > 1 else first_count

        # Build page slices
        idx = 0
        parts = []
        # First page
        take = min(first_count, total_rows - idx)
        if take > 0:
            parts.append(frame.iloc[idx:idx+take, :])
            idx += take
        # Subsequent pages
        while idx < total_rows:
            take = min(subsequent_count, total_rows - idx)
            parts.append(frame.iloc[idx:idx+take, :])
            idx += take
        return parts

    previous_was_table = False
    for a in pdf_assets:
        # Convert Markdown string to reportlab paragraphs and add them to the story
        if isinstance(a, pl.DataFrame):
            a = a.to_pandas()
        if isinstance(a, str):
            if a.startswith("You:"):
                story.append(HorizontalLine(doc.width))
                story.append(Spacer(1, 20))
            story.extend(markdown_to_paragraphs(a, styles))
            previous_was_table = False
        # Convert each DataFrame in the list to a reportlab table and add it to the story
        elif isinstance(a, pd.DataFrame) or isinstance(a, pl.DataFrame):
            # Build paginated tables with header repeating per page
            if rows_per_page is None and max_rows is None:
                # Use the provided frame as-is (no internal chunking)
                frames = [a]
            else:
                frames = _iter_paginated_frames(a, max_rows, rows_per_page)
                if not frames:
                    frames = [a]
            for i, part in enumerate(frames):
                # Ensure at least 2 columns to avoid ReportLab issues
                if isinstance(part, pd.DataFrame) and len(part.columns) == 1:
                    part = pd.concat([part, pd.Series('', name='', index=part.index)], axis='columns')
                story.append(dataframe_to_table(part, max_rows=None, max_cols=max_cols, shrink_to_fit=shrink_to_fit))
                # Force a page break between chunks on same asset except after the last chunk
                if i < len(frames) - 1:
                    story.append(PageBreak())
            # Add page break after table (so next header starts on new page)
            story.append(PageBreak())
            previous_was_table = True
        else:
            assert False, f"Unknown asset type: {type(a)}"
    
    # Build the document using the story
    doc.build(story)
    
    # Write the contents of the buffer to the output file
    if output_filename is not None:
        with open(output_filename, 'wb') as f:
            f.write(buffer.getvalue())
    
    # Return the bytes
    #return buffer.getvalue()
    return buffer.getvalue() # works for st.download() otherwise use base64.b64encode(buffer.getvalue()) # return buffer.getvalue() or base64.b64encode(buffer.getvalue())?

