"""
Base class for MLBridge Streamlit applications.
Contains standard code shared between different bridge applications.
"""

from abc import ABC, abstractmethod
import logging
import pandas as pd
import streamlit as st
import time
from typing import Dict, List, Optional, Any, Union

# todo: shouldn't other methods be abstract and initialized in main()?
class PostmortemBase(ABC):
    """Base class containing standard mlBridge code shared across applications."""
    
    def __init__(self):
        """Initialize the MLBridge application."""
        self.initialize_logging()
        # Common initialization
    
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
    
    # ---- UI Components ----
    
    def create_ui(self):
        """Creates the main UI structure."""
        self.create_sidebar()
        if not st.session_state.sql_query_mode:
            #create_tab_bar()
            if st.session_state.session_id is not None:
                self.write_report()
        self.ask_sql_query()
    

    def ShowDataFrameTable(self, df, key, query=None, show_sql_query=True, color_column=None, tooltips=None):
        """Display a DataFrame as a table with optional filtering."""
        # Implement standard dataframe display logic
        if query is not None and show_sql_query:
            st.code(query, language="sql")
        
        if df is not None and not df.is_empty():
            st.dataframe(df, use_container_width=True, key=key)
        else:
            st.info("No data available.")
    
    # ---- Data Processing Methods ----
    
    def perform_hand_augmentations(self, df, sd_productions):
        """Standard hand augmentation processing."""
        if df is None or df.is_empty():
            return df
            
        def hand_augmentation_work(df, progress, **kwargs):
            # Implementation of hand augmentation work
            pass
            
        # Progress reporting
        return df  # Return augmented dataframe
        
    def augment_df(self, df):
        """Standard dataframe augmentation."""
        if df is None or df.is_empty():
            return df
            
        # Implement standard augmentation logic
        return df
    
    def process_prompt_macros(self, sql_query):
        """Process application-specific prompt macros.
        
        This method MUST be overridden by each application.
        Args:
            sql_query: The SQL query to process
            
        Returns:
            Processed SQL query with macros expanded
        """

        replacements = {
            '{Player_Direction}': st.session_state.player_direction,
            '{Partner_Direction}': st.session_state.partner_direction,
            '{Pair_Direction}': st.session_state.pair_direction,
            '{Opponent_Pair_Direction}': st.session_state.opponent_pair_direction
        }
        
        for old, new in replacements.items():
            if new is None:
                continue
            sql_query = sql_query.replace(old, new)
        return sql_query
    
    # ---- Abstract Methods (Must Override) ----
    
    @abstractmethod
    def initialize_session_state(self):
        """Initialize application-specific session state.
        
        This method MUST be overridden by each application.
        """
        raise NotImplementedError("initialize_session_state must be implemented by subclasses")
    
    @abstractmethod
    def reset_game_data(self):
        """Initialize application-specific components.
        
        This method MUST be overridden by each application.
        """
        raise NotImplementedError("reset_game_data must be implemented by subclasses")
    
    @abstractmethod
    def initialize_website_specific(self):
        """Initialize application-specific components.
        
        This method MUST be overridden by each application.
        """
        raise NotImplementedError("initialize_website_specific must be implemented by subclasses")
    
    @abstractmethod
    def write_report(self):
        """Initialize application-specific components.
        
        This method MUST be overridden by each application.
        """
        raise NotImplementedError("write_report must be implemented by subclasses")
    
    @abstractmethod
    def ask_sql_query(self):
        """Initialize application-specific components.
        
        This method MUST be overridden by each application.
        """
        raise NotImplementedError("ask_sql_query must be implemented by subclasses")

    # ---- Main Entry Point ----
    
    def main(self):
        """Main application entry point."""
        if 'first_time' not in st.session_state:
            self.initialize_session_state()
            self.create_sidebar()
        else:
            self.create_ui()
        return
