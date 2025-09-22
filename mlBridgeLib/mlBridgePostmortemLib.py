"""
Base class for MLBridge Streamlit applications.
Contains standard code shared between different bridge applications.
"""

from abc import ABC, abstractmethod
import logging
import os
import polars as pl
import streamlit as st
import streamlit_chat
from streamlit_extras.bottom_container import bottom
from stqdm import stqdm
import pathlib
import duckdb
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union

# todo: shouldn't other methods be abstract and initialized in main()?
class PostmortemBase(ABC):
    """Base class containing standard mlBridge code shared across applications."""
    
    def __init__(self):
        """Initialize the MLBridge application."""
        self.initialize_logging()
        # Common initialization will be done in initialize_session_state
    
    def initialize_logging(self):
        """Set up logging configuration."""
        from mlBridgeLib.logging_config import setup_logger
        self.logger = setup_logger(__name__)
    
    def print_to_log_info(self, *args):
        """Log information messages."""
        self.print_to_log(logging.INFO, *args)
    
    def print_to_log_debug(self, *args):
        """Log debug messages."""
        self.print_to_log(logging.DEBUG, *args)
    
    def print_to_log(self, level, *args):
        """Log messages at specified level."""
        self.logger.log(level, ' '.join(str(arg) for arg in args))
    
    # ---- Database Connection Management ----
    
    def get_session_duckdb_connection(self):
        """Get or create a DuckDB connection for the current session
        
        Returns:
            duckdb.DuckDBPyConnection: Session-specific DuckDB connection
        """
        # Prefer the app's connection key to avoid duplicate connections
        if 'db_connection' in st.session_state and st.session_state.db_connection is not None:
            return st.session_state.db_connection

        # Fall back to legacy key 'con' if present, and normalize to 'db_connection'
        if 'con' in st.session_state and st.session_state.con is not None:
            st.session_state.db_connection = st.session_state.con
            return st.session_state.con

        # Otherwise create a new connection and set both keys
        con = duckdb.connect()  # In-memory database per session
        st.session_state.db_connection = con
        st.session_state.con = con
        print(f"Created new DuckDB connection for session")
        return con
    
    # ---- UI Components ----
    
    def create_ui(self):
        """Creates the main UI structure."""
        self.create_sidebar()
        if not st.session_state.sql_query_mode:
            #create_tab_bar()
            if st.session_state.session_id is not None:
                self.write_report()
        self.ask_sql_query()
    
    def ShowDataFrameTable(self, df: Optional[pl.DataFrame], key: str, query: str = 'SELECT * FROM self', show_sql_query: bool = True, height_rows: int = 4) -> Optional[pl.DataFrame]:
        """Display a DataFrame table in Streamlit with optional SQL query execution
        
        Args:
            df: The Polars DataFrame to display (None to use existing registered table)
            key: Unique key for the Streamlit component
            query: SQL query to execute on the DataFrame
            show_sql_query: Whether to display the SQL query text
            height_rows: Number of rows to display in the table (default: 4)
            
        Returns:
            Result DataFrame from SQL query, or None if query failed
        """
        if show_sql_query and st.session_state.show_sql_query:
            st.text(f"SQL Query: {query}")

        # Normalize/prepend FROM "self" to avoid Python replacement scans on the instance variable `self`
        if 'from "self"' not in query.lower():
            query = query.replace('FROM self', 'FROM "self"')
            query = query.replace('from self', 'FROM "self"')
        if 'from "self"' not in query.lower():
            query = 'FROM "self" ' + query

        try:
            con = self.get_session_duckdb_connection()
            # Validate the single source of truth is registered by the app
            try:
                con.execute('DESCRIBE "self"').fetchall()
            except Exception:
                st.info("Data is loading... Please wait for processing to complete.")
                return None

            result_df = con.execute(query).pl()
            if show_sql_query and st.session_state.show_sql_query:
                st.text(f"Result is a dataframe of {len(result_df)} rows.")
            
            # Import streamlitlib for display
            sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))
            import streamlitlib
            streamlitlib.ShowDataFrameTable(result_df, key, height_rows=height_rows)
        except Exception as e:
            st.error(f"duckdb exception: error:{e} query:{query}")
            return None
        
        return result_df
    
    # ---- Data Processing Methods ----
    
    def perform_hand_augmentations_queue(self, augmenter_instance, hand_augmentation_work):
        """Perform hand augmentations queue processing
        
        Args:
            augmenter_instance: The augmenter instance calling this method
            hand_augmentation_work: Work item for hand augmentation processing
        """
        # Import streamlitlib for queued work processing
        sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))
        import streamlitlib
        return streamlitlib.perform_queued_work(augmenter_instance, hand_augmentation_work, "Hand analysis")
        
    def augment_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Augment DataFrame with additional bridge analysis data
        
        Args:
            df: Input DataFrame containing bridge game data
            
        Returns:
            Augmented DataFrame with additional analysis columns
        """
        if df is None or df.is_empty():
            return df
            
        with st.spinner('Augmenting data...'):
            # Import augmentation library
            sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))
            from mlBridgeLib.mlBridgeAugmentLib import AllAugmentations
            
            augmenter = AllAugmentations(
                df, None, 
                sd_productions=st.session_state.single_dummy_sample_count,
                progress=st.progress(0),
                lock_func=self.perform_hand_augmentations_queue
            )
            df, hrs_cache_df = augmenter.perform_all_augmentations()
        return df
    
    def process_prompt_macros(self, sql_query: str) -> str:
        """Process SQL query macros and replace them with actual values
        
        Args:
            sql_query: Input SQL query string with macros
            
        Returns:
            Processed SQL query with macros replaced
        """
        replacements = {
            '{Player_Direction}': getattr(st.session_state, 'player_direction', None),
            '{Partner_Direction}': getattr(st.session_state, 'partner_direction', None),
            '{Pair_Direction}': getattr(st.session_state, 'pair_direction', None),
            '{Opponent_Pair_Direction}': getattr(st.session_state, 'opponent_pair_direction', None)
        }
        
        for old, new in replacements.items():
            if new is None:
                continue
            sql_query = sql_query.replace(old, new)
        return sql_query
    
    # ---- Configuration Management ----
    
    def read_configs(self) -> Dict[str, Any]:
        """Read configuration files and return configuration dictionary
        
        Returns:
            Dictionary containing configuration settings
        """
        player_id = getattr(st.session_state, 'player_id', 'unknown')
        
        st.session_state.default_favorites_file = pathlib.Path('default.favorites.json')
        st.session_state.player_id_custom_favorites_file = pathlib.Path(f'favorites/{player_id}.favorites.json')
        st.session_state.debug_favorites_file = pathlib.Path('favorites/debug.favorites.json')

        # Load default favorites
        if st.session_state.default_favorites_file.exists():
            with open(st.session_state.default_favorites_file, 'r') as f:
                favorites = json.load(f)
            st.session_state.favorites = favorites
        else:
            st.session_state.favorites = None

        # Load player-specific favorites
        if st.session_state.player_id_custom_favorites_file.exists():
            with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
                player_id_favorites = json.load(f)
            st.session_state.player_id_favorites = player_id_favorites
        else:
            st.session_state.player_id_favorites = None

        # Load debug favorites
        if st.session_state.debug_favorites_file.exists():
            with open(st.session_state.debug_favorites_file, 'r') as f:
                debug_favorites = json.load(f)
            st.session_state.debug_favorites = debug_favorites
        else:
            st.session_state.debug_favorites = None

        # Validate favorites consistency
        if st.session_state.favorites and 'missing_in_summarize' not in st.session_state:
            try:
                summarize_prompts = st.session_state.favorites['Buttons']['Summarize']['prompts']
                vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']

                # Process the keys to ignore leading '@'
                st.session_state.summarize_keys = {p.lstrip('@') for p in summarize_prompts}
                st.session_state.vetted_keys = set(vetted_prompts.keys())

                # Find items in summarize_prompts but not in vetted_prompts
                st.session_state.missing_in_vetted = st.session_state.summarize_keys - st.session_state.vetted_keys
                if len(st.session_state.missing_in_vetted) > 0:
                    print(f"Warning: {st.session_state.missing_in_vetted} not in {st.session_state.vetted_keys}")

                # Find items in vetted_prompts but not in summarize_prompts
                st.session_state.missing_in_summarize = st.session_state.vetted_keys - st.session_state.summarize_keys

                print("\nItems in Vetted_Prompts but not in Summarize.prompts:")
                for item in st.session_state.missing_in_summarize:
                    print(f"- {item}: {vetted_prompts[item]['title']}")
            except (KeyError, TypeError) as e:
                print(f"Warning: Error processing favorites configuration: {e}")
        
        return getattr(st.session_state, 'favorites', {})
    
    # ---- Common Session State Management ----
    
    def initialize_common_session_state(self):
        """Initialize common session state variables used across all applications."""
        st.set_page_config(layout="wide")
        
        # Platform-specific path handling
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        else:
            pathlib.WindowsPath = pathlib.PosixPath
        
        # Common session state defaults
        streamlit_envs = os.getenv('STREAMLIT_ENV', '').split(',') if os.getenv('STREAMLIT_ENV') else []
        first_time_defaults = {
            'first_time': True,
            'single_dummy_sample_count': 10,
            'show_sql_query': 'debug_mode' in streamlit_envs or 'sql_query_mode' in streamlit_envs,
            'use_historical_data': False,
            'do_not_cache_df': True,
            'con_register_name': 'self',
            'main_section_container': st.empty(),
            'app_datetime': datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'current_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sql_query_mode': False,
            'sql_queries': [],
            'pdf_assets': [],
            'vetted_prompts': [],
            'analysis_started': False,
            'debug_mode': 'debug_mode' in streamlit_envs,
        }
        
        for key, value in first_time_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Initialize session-specific DuckDB connection
        self.get_session_duckdb_connection()
        
        # Import and setup streamlitlib
        sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))
        sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))
        import streamlitlib
        streamlitlib.widen_scrollbars()
    
    def reset_common_game_data(self):
        """Reset common game data to default values."""
        # Default values for session state variables
        reset_defaults = {
            'game_description_default': None,
            'group_id_default': None,
            'session_id_default': None,
            'section_name_default': None,
            'player_id_default': None,
            'partner_id_default': None,
            'player_name_default': None,
            'partner_name_default': None,
            'player_direction_default': None,
            'partner_direction_default': None,
            'pair_id_default': None,
            'pair_direction_default': None,
            'opponent_pair_direction_default': None,
        }
        
        # Initialize default values if not already set
        for key, value in reset_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Initialize additional session state variables that depend on defaults
        reset_session_vars = {
            'df': None,
            'game_description': st.session_state.game_description_default,
            'group_id': st.session_state.group_id_default,
            'session_id': st.session_state.session_id_default,
            'section_name': st.session_state.section_name_default,
            'player_id': st.session_state.player_id_default,
            'partner_id': st.session_state.partner_id_default,
            'player_name': st.session_state.player_name_default,
            'partner_name': st.session_state.partner_name_default,
            'player_direction': st.session_state.player_direction_default,
            'partner_direction': st.session_state.partner_direction_default,
            'pair_id': st.session_state.pair_id_default,
            'pair_direction': st.session_state.pair_direction_default,
            'opponent_pair_direction': st.session_state.opponent_pair_direction_default,
            'analysis_started': False,
            'vetted_prompts': [],
            'pdf_assets': [],
            'sql_query_mode': False,
            'sql_queries': [],
            'game_urls_d': {},
            'tournament_session_urls_d': {},
            'current_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        for key, value in reset_session_vars.items():
            st.session_state[key] = value
    
    # ---- Standard Report Generation ----
    
    def write_standard_report(self):
        """Generate standard postmortem report using favorites configuration."""
        if not st.session_state.favorites:
            st.error("No favorites configuration found. Cannot generate report.")
            return
            
        st.session_state.main_section_container = st.container(border=True)
        with st.session_state.main_section_container:
            # Report header information
            report_title = f"Bridge Game Postmortem Report"
            if hasattr(st.session_state, 'player_name') and st.session_state.player_name:
                report_title += f" Personalized for {st.session_state.player_name}"
            
            report_creator = f"Created by https://{getattr(st.session_state, 'game_name', 'bridge')}.postmortem.chat"
            
            # Event information
            event_info_parts = []
            if hasattr(st.session_state, 'organization_name') and st.session_state.organization_name:
                event_info_parts.append(st.session_state.organization_name)
            if hasattr(st.session_state, 'game_description') and st.session_state.game_description:
                event_info_parts.append(st.session_state.game_description)
            if hasattr(st.session_state, 'session_id') and st.session_state.session_id:
                event_info_parts.append(f"(event id {st.session_state.session_id})")
            
            report_event_info = " ".join(event_info_parts) if event_info_parts else "Bridge Game"
            
            # Game results webpage
            report_game_results_webpage = ""
            if hasattr(st.session_state, 'game_url') and st.session_state.game_url:
                report_game_results_webpage = f"Results Page: {st.session_state.game_url}"
            
            # Player match information
            match_info_parts = []
            if hasattr(st.session_state, 'pair_id') and st.session_state.pair_id:
                match_info_parts.append(f"Your pair was {st.session_state.pair_id}")
            if hasattr(st.session_state, 'pair_direction') and st.session_state.pair_direction:
                match_info_parts.append(st.session_state.pair_direction)
            if hasattr(st.session_state, 'section_name') and st.session_state.section_name:
                match_info_parts.append(f"in section {st.session_state.section_name}")
            if hasattr(st.session_state, 'player_direction') and st.session_state.player_direction:
                match_info_parts.append(f"You played {st.session_state.player_direction}")
            if hasattr(st.session_state, 'partner_name') and st.session_state.partner_name:
                match_info_parts.append(f"Your partner was {st.session_state.partner_name}")
            if hasattr(st.session_state, 'partner_direction') and st.session_state.partner_direction:
                match_info_parts.append(f"who played {st.session_state.partner_direction}")
            
            report_your_match_info = ". ".join(match_info_parts) if match_info_parts else ""
            
            # Display report header
            st.markdown('<div style="height: 50px;"><a name="top-of-report"></a></div>', unsafe_allow_html=True)
            st.markdown(f"### {report_title}")
            st.markdown(f"##### {report_creator}")
            if report_event_info:
                st.markdown(f"#### {report_event_info}")
            if report_game_results_webpage:
                st.markdown(f"##### {report_game_results_webpage}")
            if report_your_match_info:
                st.markdown(f"#### {report_your_match_info}")
            
            # Initialize PDF assets
            pdf_assets = st.session_state.pdf_assets
            pdf_assets.clear()
            pdf_assets.append(f"# {report_title}")
            pdf_assets.append(f"#### {report_creator}")
            if report_event_info:
                pdf_assets.append(f"### {report_event_info}")
            if report_game_results_webpage:
                pdf_assets.append(f"#### {report_game_results_webpage}")
            if report_your_match_info:
                pdf_assets.append(f"### {report_your_match_info}")
            
            # Generate report sections from favorites
            try:
                st.session_state.button_title = 'Summarize'
                selected_button = st.session_state.favorites['Buttons'][st.session_state.button_title]
                vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']
                sql_query_count = 0
                
                for stats in stqdm(selected_button['prompts'], desc='Creating personalized report...'):
                    assert stats[0] == '@', stats
                    stat = vetted_prompts[stats[1:]]
                    for i, prompt in enumerate(stat['prompts']):
                        if 'sql' in prompt and prompt['sql']:
                            if i == 0:
                                streamlit_chat.message(
                                    f"Morty: {stat['help']}", 
                                    key=f'morty_sql_query_{sql_query_count}', 
                                    logo=getattr(st.session_state, 'assistant_logo', None)
                                )
                                pdf_assets.append(f"### {stat['help']}")
                            
                            prompt_sql = prompt['sql']
                            sql_query = self.process_prompt_macros(prompt_sql)
                            query_df = self.ShowDataFrameTable(
                                st.session_state.df, 
                                query=sql_query, 
                                key=f'sql_query_{sql_query_count}'
                            )
                            if query_df is not None:
                                pdf_assets.append(query_df)
                            sql_query_count += 1
            except (KeyError, TypeError) as e:
                st.error(f"Error generating report from favorites: {e}")
            
            # Go to top button
            st.markdown('''
                <div style="text-align: center; margin: 20px 0;">
                    <a href="#top-of-report" style="text-decoration: none;">
                        <button style="padding: 8px 16px; background-color: #ff4b4b; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">
                            Go to top of report
                        </button>
                    </a>
                </div>
            ''', unsafe_allow_html=True)

        # PDF download button
        if hasattr(st.session_state, 'pdf_link') and st.session_state.pdf_link:
            try:
                import streamlitlib
                if st.session_state.pdf_link.download_button(
                    label="Download Personalized Report PDF",
                    data=streamlitlib.create_pdf(
                        st.session_state.pdf_assets, 
                        title=f"Bridge Game Postmortem Report Personalized for {getattr(st.session_state, 'player_id', 'Player')}"
                    ),
                    file_name=f"{getattr(st.session_state, 'session_id', 'session')}-{getattr(st.session_state, 'player_id', 'player')}-morty.pdf",
                    disabled=len(st.session_state.pdf_assets) == 0,
                    mime='application/octet-stream',
                    key='personalized_report_download_button'
                ):
                    st.warning('Personalized report downloaded.')
            except Exception as e:
                print(f"Error creating PDF download: {e}")
    
    def ask_standard_sql_query(self):
        """Handle standard SQL query input and display results."""
        if st.session_state.show_sql_query:
            st.chat_input(
                'Enter a SQL query e.g. SELECT PBN, Contract, Result, N, S, E, W', 
                key='main_prompt_chat_input', 
                on_submit=self.chat_input_on_submit
            )
    
    def chat_input_on_submit(self):
        """Handle chat input submission and process SQL queries."""
        prompt = st.session_state.main_prompt_chat_input
        sql_query = self.process_prompt_macros(prompt)
        if not st.session_state.sql_query_mode:
            st.session_state.sql_query_mode = True
            st.session_state.sql_queries.clear()
        st.session_state.sql_queries.append((prompt, sql_query))
        st.session_state.main_section_container = st.empty()
        st.session_state.main_section_container = st.container()
        with st.session_state.main_section_container:
            for i, (prompt, sql_query) in enumerate(st.session_state.sql_queries):
                self.ShowDataFrameTable(st.session_state.df, query=sql_query, key=f'user_query_main_doit_{i}')
    
    # ---- Abstract Methods (Must Override) ----
    
    @abstractmethod
    def initialize_session_state(self):
        """Initialize application-specific session state.
        
        This method MUST be overridden by each application.
        Should call initialize_common_session_state() first.
        """
        raise NotImplementedError("initialize_session_state must be implemented by subclasses")
    
    @abstractmethod
    def reset_game_data(self):
        """Reset application-specific game data.
        
        This method MUST be overridden by each application.
        Should call reset_common_game_data() first.
        """
        raise NotImplementedError("reset_game_data must be implemented by subclasses")
    
    @abstractmethod
    def initialize_website_specific(self):
        """Initialize application-specific components.
        
        This method MUST be overridden by each application.
        """
        raise NotImplementedError("initialize_website_specific must be implemented by subclasses")
    
    @abstractmethod
    def create_sidebar(self):
        """Create application-specific sidebar.
        
        This method MUST be overridden by each application.
        """
        raise NotImplementedError("create_sidebar must be implemented by subclasses")
    
    def write_report(self):
        """Generate postmortem report. Can be overridden for custom reports."""
        self.write_standard_report()
    
    def ask_sql_query(self):
        """Handle SQL query interface. Can be overridden for custom behavior."""
        self.ask_standard_sql_query()

    # ---- Main Entry Point ----
    
    def main(self):
        """Main application entry point."""
        if 'first_time' not in st.session_state:
            self.initialize_session_state()
            self.create_sidebar()
        else:
            self.create_ui()
        return
