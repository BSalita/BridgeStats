# contains functions to augment df with additional columns
# mostly polars functions
# some functionality is similar to mlBridgeAcblLib.py which operates on a single game download, not all games in the db.

# todo:
# DD_Score_\d_[CDHSN]_[NESW] has reversed strain and direction columns compared to other column naming conventions.
# optimize slow functions: _create_dd_columns() and calculate_final_scores()
# since some columns can be derived from other columns, we should assert that input df has at least one column in each group of mutually derivable columns.
# assert that column names don't exist in df.columns for all column creation functions.
# refactor when should *_Dcl columns be created? At end of each func, class, class of it's own?
# if a column already exists, print a message and skip creation.
# if a column already exists, generate a new column and assert that the new column is the same as the existing column.
# print column names and dtypes for all columns generated and skipped.
# print a list of mandatory columns that must be present in df.columns. many columns can be derived from other columns e.g. scoring columns.
# for each augmentation function, validate that all columns are properly generated.
# create a function which validates that all needed for the class can be derived from the input df.
# create a function which validates that all columns for the class are generated.
# create a function which validate columns for every column generated.
# use .pipe() to chain other lambdas as is done in _add_trick_probabilities()?
# should *_Declarer be the pattern or omit _Declarer where declarer would be implied?
# change *_Declarer to *_Dcl?
# looks like Friday simultaneous doesn't have Contracts? 8-Aug-2025 did not. Is it posted with a 7+ day delay?
# create a function which creates a column of the weighted moving average of (double dummy tricks for declarer's contract minus tricks taken) per start of session. Each row of the session has the same start of session value.
# Rename to title case for column names. e.g. session_id to Session_ID
# change Number_Declarer to Declarer or Declarer_ID or Player_ID_Declarer. Same with Declarer_Name or Player_Name_Declarer. Same with OnLead or OnLead_ID or Player_ID_OnLead.

import polars as pl
import numpy as np
import warnings
from collections import defaultdict
from typing import Optional, Union, Callable, Type, Dict, List, Tuple, Any
import time


import endplay # for __version__
from endplay.parsers import pbn, lin, json
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
from endplay.dds import calc_dd_table, calc_all_tables, par
from endplay.dealer import generate_deals

import mlBridgeLib.mlBridgeLib as mlBridgeLib
from mlBridgeLib.mlBridgeLib import (
    NESW, SHDC, NS_EW,
    PlayerDirectionToPairDirection,
    NextPosition,
    PairDirectionToOpponentPairDirection,
    score
)
from logging_config import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__)


# todo: use versions in mlBridgeLib
VulToEndplayVul_d = { # convert mlBridgeLib Vul to endplay Vul
    'None':Vul.none,
    'Both':Vul.both,
    'N_S':Vul.ns,
    'E_W':Vul.ew
}


DealerToEndPlayDealer_d = { # convert mlBridgeLib dealer to endplay dealer
    'N':Player.north,
    'E':Player.east,
    'S':Player.south,
    'W':Player.west
}

declarer_to_LHO_d = {
    None:None,
    'N':'E',
    'E':'S',
    'S':'W',
    'W':'N'
}


declarer_to_dummy_d = {
    None:None,
    'N':'S',
    'E':'W',
    'S':'N',
    'W':'E'
}


declarer_to_RHO_d = {
    None:None,
    'N':'W',
    'E':'N',
    'S':'E',
    'W':'S'
}

# Pre-generated column name constants to avoid repeated computation
DD_SCORE_COLUMNS = [f"DD_Score_{level}{strain}_{direction}" 
                   for level in range(1, 8)
                   for strain in 'CDHSN'  
                   for direction in 'NESW'] + [f"DD_Score_{level}{strain}_{pair}"
                   for level in range(1, 8)
                   for strain in 'CDHSN'  
                   for pair in ['NS', 'EW']]

EV_SCORE_COLUMNS = [f'EV_{pair}_{declarer}_{strain}_{level}_{vul}' 
                   for pair in ['NS','EW'] 
                   for declarer in pair 
                   for strain in 'SHDCN' 
                   for level in range(1,8)
                   for vul in ['V', 'NV']]

DD_COLUMNS = [f'DD_{direction}_{suit}' for suit in 'SHDCN' for direction in 'NESW']

PROB_COLUMNS = [f'Probs_{pair}_{declarer}_{strain}_{i}' 
               for pair in ['NS', 'EW'] 
               for declarer in 'NESW' 
               for strain in 'CDHSN' 
               for i in range(14)]


# =============================================================================
# SECTION 1: FOUNDATION FUNCTIONS
# =============================================================================
# Used by: DealAugmenter and early initialization
# Purpose: PBN parsing, basic hand setup, dealer/vulnerability initialization
# Temporal order: First - these create the basic hand and deal structure
# =============================================================================

def parse_pbn_to_hands(df: pl.DataFrame) -> pl.DataFrame:    
    """Create hand strings for each seat from PBN.

    Purpose:
    - Parse `PBN` into four seat-specific columns `Hand_N`, `Hand_E`, `Hand_S`, `Hand_W`.

    Parameters:
    - df: Polars DataFrame.

    Input columns:
    - `PBN`: string like 'N:... ... ... ...'.

    Output columns:
    - `Hand_N`, `Hand_E`, `Hand_S`, `Hand_W` (pl.String).

    Returns:
    - DataFrame with added hand columns if missing.
    """
    # create 'Hand_[NESW]' columns of type pl.String from 'PBN'
    if 'Hand_N' not in df.columns:
        for i, direction in enumerate('NESW'):
            df = df.with_columns([
                pl.col('PBN')   
              .str.slice(2)
              .str.split(' ')
              .list.get(i)
              .alias(f'Hand_{direction}')
    ])
    return df


def parse_pbn_to_hands_list(df: pl.DataFrame) -> pl.DataFrame:
    """Create nested `Hands` list column from `PBN`.

    Purpose:
    - Produce `Hands` as list-of-hands where each hand is [S,H,D,C] strings.

    Parameters:
    - df: Polars DataFrame.

    Input columns:
    - `PBN`.

    Output columns:
    - `Hands`: pl.List(pl.List(pl.String)).

    Returns:
    - DataFrame with `Hands` added if missing.
    """
    # create 'Hands' column of type pl.List(pl.List(pl.String)) from 'PBN'
    if 'Hands' not in df.columns:
        df = df.with_columns([  
            pl.col('PBN')
           .str.slice(2)
           .str.split(' ')
           .list.eval(pl.element().str.split('.'), parallel=True)
           .alias('Hands')
        ])
    return df


def extract_suits_by_seat(df: pl.DataFrame) -> pl.DataFrame:
    """Create suit strings by seat and suit from `Hand_[NESW]`.

    Purpose:
    - Split `Hand_[NESW]` into `Suit_[NESW]_[SHDC]` columns.

    Parameters:
    - df: Polars DataFrame.

    Input columns:
    - `Hand_N`, `Hand_E`, `Hand_S`, `Hand_W`.

    Output columns:
    - `Suit_[NESW]_[SHDC]` (pl.String) for all seats and suits.

    Returns:
    - DataFrame with suit columns added if missing.
    """
    # Create 'Suit_[NESW]_[SHDC]' columns of type pl.String
    if 'Suit_N_C' not in df.columns:
        for d in 'NESW':
            for i, s in enumerate('SHDC'):
                df = df.with_columns([
                    pl.col(f'Hand_{d}')
                    .str.split('.')
                   .list.get(i)
                   .alias(f'Suit_{d}_{s}')
          ])
    return df


# One Hot Encoded into binary string
def encode_hands_binary(hands_bin: list[list[Tuple[Optional[str], Optional[str]]]]) -> defaultdict[str, list[Any]]:
    """One-hot encode hands into binary-like fields (legacy helper).

    Purpose:
    - Convert nested hand data structure into binary-encoded format for machine learning
    - Extract directional hand data and organize by position (N/E/S/W)
    - Provide legacy support for binary encoding of bridge hands

    Parameters:
    - hands_bin: nested list structure with seat tuples containing hand data.

    Input columns:
    - None (operates on raw nested list data structure, not DataFrame columns).

    Output columns:
    - None (returns dictionary structure, not DataFrame columns).

    Returns:
    - defaultdict mapping keys like 'HB_N' -> list of encoded values.
    """
    # Vectorized approach: transpose the data structure to avoid nested loops
    handsbind = defaultdict(list)
    
    # Pre-validate all data 
    for h in hands_bin:
        for nesw in h:
            assert nesw[0] is not None and nesw[1] is not None
    
    # Efficient extraction using list comprehensions and zip
    for i, direction in enumerate(NESW):
        handsbind[f'HB_{direction}'] = [h[i][0] for h in hands_bin]
    
    return handsbind


def add_dealer_column(df: pl.DataFrame) -> pl.DataFrame:
    """Create Dealer column from Board number using vectorized arithmetic and mapping.

    Purpose:
    - Calculate the dealer for each board using standard bridge rotation
    - Apply modular arithmetic to map board numbers to dealer directions
    - Use efficient Polars vectorized operations

    Parameters:
    - df: DataFrame containing board number information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Dealer' column

    Input columns:
    - 'Board': Board number (typically 1-based numbering)

    Output columns:
    - 'Dealer': Single character dealer direction (N/E/S/W)

    Notes:
    - Uses standard bridge dealer rotation: Board 1=N, 2=E, 3=S, 4=W, then repeats
    """
    dealer_idx = (pl.col('Board') - 1) % 4
    return df.with_columns(
        dealer_idx.replace_strict({0: 'N', 1: 'E', 2: 'S', 3: 'W'}).alias('Dealer')
    )


def encode_vulnerability(df: pl.DataFrame) -> pl.DataFrame:
    """Create numeric vulnerability encoding from string vulnerability using vectorized replacement.

    Purpose:
    - Convert string vulnerability notation to numeric codes for calculations
    - Enable efficient vulnerability-based filtering and logic
    - Standardize vulnerability representation across the system

    Parameters:
    - df: DataFrame containing string vulnerability information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'iVul' column

    Input columns:
    - 'Vul': String vulnerability ('None', 'N_S', 'E_W', 'Both')

    Output columns:
    - 'iVul': Numeric vulnerability code (0=None, 1=N_S, 2=E_W, 3=Both)
    """
    mapping = {'None': 0, 'N_S': 1, 'E_W': 2, 'Both': 3}
    return df.with_columns(
        pl.col('Vul').replace_strict(mapping).cast(pl.UInt8).alias('iVul')
    )


def derive_numeric_vulnerability_from_board(df: pl.DataFrame) -> pl.DataFrame:
    """Create numeric vulnerability from board number using standard bridge vulnerability schedule.

    Purpose:
    - Calculate vulnerability based on board number using official bridge rules
    - Apply 16-board vulnerability cycle used in duplicate bridge
    - Generate consistent vulnerability patterns for tournaments

    Parameters:
    - df: DataFrame containing board number information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'iVul' column

    Input columns:
    - 'Board': Board number (typically 1-based)

    Output columns:
    - 'iVul': Numeric vulnerability code (0=None, 1=N_S, 2=E_W, 3=Both)

    Notes:
    - Implements standard 16-board vulnerability cycle used in duplicate bridge
    - Board numbers cycle through vulnerability pattern every 16 boards
    """
    def board_number_to_vul(bn: int) -> int:
        bn -= 1
        return range(bn//4, bn//4+4)[bn & 3] & 3
    
    return df.with_columns(
        pl.col('Board')
        .map_elements(board_number_to_vul, return_dtype=pl.UInt8)
        .alias('iVul')
    )


def decode_vulnerability(df: pl.DataFrame) -> pl.DataFrame:
    """Create string vulnerability from numeric vulnerability using vectorized replacement.

    Purpose:
    - Convert numeric vulnerability codes back to readable string format
    - Enable human-readable vulnerability display and export
    - Reverse the encoding process for data presentation

    Parameters:
    - df: DataFrame containing numeric vulnerability information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Vul' column

    Input columns:
    - 'iVul': Numeric vulnerability code (0=None, 1=N_S, 2=E_W, 3=Both)

    Output columns:
    - 'Vul': String vulnerability ('None', 'N_S', 'E_W', 'Both')
    """
    mapping = {0: 'None', 1: 'N_S', 2: 'E_W', 3: 'Both'}
    return df.with_columns(
        pl.col('iVul').replace_strict(mapping).alias('Vul')
    )


def add_pair_vulnerability_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Create pair-specific vulnerability flags from general vulnerability column.

    Purpose:
    - Convert general vulnerability status to specific pair vulnerability flags
    - Enable pair-specific vulnerability logic and calculations
    - Create boolean flags for efficient vulnerability filtering

    Parameters:
    - df: DataFrame containing general vulnerability information

    Returns:
    - pl.DataFrame: Input DataFrame with added pair vulnerability columns

    Input columns:
    - 'Vul': String vulnerability ('None', 'N_S', 'E_W', 'Both')

    Output columns:
    - 'Vul_NS': Boolean flag indicating if North-South is vulnerable
    - 'Vul_EW': Boolean flag indicating if East-West is vulnerable
    """
    return df.with_columns([
        pl.Series('Vul_NS', df['Vul'].is_in(['N_S','Both']), pl.Boolean),
        pl.Series('Vul_EW', df['Vul'].is_in(['E_W','Both']), pl.Boolean)
    ])


# =============================================================================
# SECTION 2: HAND ANALYSIS FUNCTIONS  
# =============================================================================
# Used by: HandAugmenter
# Purpose: Card analysis, HCP, suit lengths, quick tricks, hand evaluation
# Temporal order: Second - these analyze the parsed hands for bridge metrics
# =============================================================================

# generic function to augment metrics by suits
def expand_metric_by_suits(metrics: pl.DataFrame, metric: str, dtype: pl.DataType = pl.UInt8) -> pl.DataFrame:
    """Expand a nested metric column into seat- and suit-specific columns.

    Purpose:
    - Given a column `metric` with nested lists, derive per-seat and per-suit scalar columns.

    Parameters:
    - metrics: input DataFrame containing the metric column.
    - metric: name of the nested metric column.
    - dtype: target dtype for derived columns.

    Input columns:
    - `metric` (nested list structure positionally indexed).

    Output columns:
    - `{metric}_{N|E|S|W}` and `{metric}_{N|E|S|W}_{S|H|D|C}`.

    Returns:
    - DataFrame with added derived columns.
    """
    # Create direction-specific columns using list access
    for d, direction in enumerate(NESW):
        # Extract direction-specific values using list indexing
        direction_expr = pl.col(metric).list.get(1).list.get(d).list.get(0).cast(dtype).alias('_'.join([metric, direction]))
        
        # Create suit-specific columns
        suit_exprs = []
        for s, suit in enumerate(SHDC):
            suit_expr = pl.col(metric).list.get(1).list.get(d).list.get(1).list.get(s).cast(dtype).alias('_'.join([metric, direction, suit]))
            suit_exprs.append(suit_expr)
        
        metrics = metrics.with_columns([direction_expr] + suit_exprs)
    
    # Create pair direction columns by summing individual directions
    for direction in NS_EW:
        pair_expr = (
            pl.col('_'.join([metric, direction[0]])) + 
            pl.col('_'.join([metric, direction[1]]))
        ).cast(dtype).alias('_'.join([metric, direction]))
        
        # Create pair suit columns
        pair_suit_exprs = []
        for s, suit in enumerate(SHDC):
            pair_suit_expr = (
                pl.col('_'.join([metric, direction[0], suit])) + 
                pl.col('_'.join([metric, direction[1], suit]))
            ).cast(dtype).alias('_'.join([metric, direction, suit]))
            pair_suit_exprs.append(pair_suit_expr)
        
        metrics = metrics.with_columns([pair_expr] + pair_suit_exprs)
    
    return metrics


def add_card_presence_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Derive binary card presence columns from Suit strings.

    Purpose:
    - Create binary indicators for every card in every hand
    - Enable card-specific analysis and calculations
    - Foundation for HCP and other card-based metrics

    Parameters:
    - df: DataFrame containing suit string columns

    Returns:
    - DataFrame with added card presence columns

    Input columns:
    - `Suit_[NESW]_[SHDC]`: Suit strings like 'AKQ..'

    Output columns:
    - `C_[NESW][SHDC][AKQJT98765432]`: Boolean presence for each card
    """
    df = df.with_columns([
        pl.col(f'Suit_{direction}_{suit}').str.contains(rank).alias(f'C_{direction}{suit}{rank}')
        for direction in 'NESW'
        for suit in 'SHDC'
        for rank in 'AKQJT98765432'
    ])
    return df


def add_hcp_from_cards(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate High Card Points (HCP) for bridge hands using vectorized operations.
    
    Purpose:
    - Compute standard bridge HCP values (A=4, K=3, Q=2, J=1)
    - Generate HCP totals by direction, suit, and partnership
    - Create comprehensive HCP analysis for hand evaluation

    Parameters:
    - df: DataFrame with card presence columns

    Returns:
    - DataFrame with added HCP columns

    Input columns:
    - `C_[NESW][SHDC][AKQJ]`: Boolean card presence for honors

    Output columns:
    - `HCP_[NESW]_[SHDC]`: HCP by direction and suit
    - `HCP_[NESW]`: Total HCP by direction
    - `HCP_(NS|EW)`: Partnership HCP totals
    - `HCP_(NS|EW)_[SHDC]`: Partnership HCP by suit
    """
    hcp_d = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}

    # Step 1: Calculate HCP for each direction and suit
    hcp_suit_expr = [
        pl.sum_horizontal([pl.col(f'C_{d}{s}{r}').cast(pl.UInt8) * v for r, v in hcp_d.items()]).alias(f'HCP_{d}_{s}')
        for d in 'NESW' for s in 'SHDC'
    ]
    df = df.with_columns(hcp_suit_expr)

    # Step 2: Calculate total HCP for each direction
    hcp_direction_expr = [
        pl.sum_horizontal([pl.col(f'HCP_{d}_{s}') for s in 'SHDC']).alias(f'HCP_{d}')
        for d in 'NESW'
    ]
    df = df.with_columns(hcp_direction_expr)

    # Step 3: Calculate HCP for partnerships
    hcp_partnership_expr = [
        (pl.col('HCP_N') + pl.col('HCP_S')).alias('HCP_NS'),
        (pl.col('HCP_E') + pl.col('HCP_W')).alias('HCP_EW')
    ]
    df = df.with_columns(hcp_partnership_expr)

    # Step 4: Calculate HCP for partnerships by suit
    hcp_partnership_suit_expr = [
        (pl.col(f'HCP_N_{s}') + pl.col(f'HCP_S_{s}')).alias(f'HCP_NS_{s}')
        for s in 'SHDC'
    ] + [
        (pl.col(f'HCP_E_{s}') + pl.col(f'HCP_W_{s}')).alias(f'HCP_EW_{s}')
        for s in 'SHDC'
    ]
    df = df.with_columns(hcp_partnership_suit_expr)

    return df


def add_quick_tricks(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate Quick Tricks for bridge hands using vectorized pattern matching.
    
    Purpose:
    - Compute defensive quick tricks based on honor combinations
    - Generate QT totals by direction, suit, and partnership
    - Provide defensive strength assessment for hand evaluation

    Parameters:
    - df: DataFrame with suit string columns

    Returns:
    - DataFrame with added Quick Tricks columns

    Input columns:
    - `Suit_[NESW]_[SHDC]`: Suit strings for pattern matching

    Output columns:
    - `QT_[NESW]_[SHDC]`: Quick tricks by direction and suit
    - `QT_[NESW]`: Total quick tricks by direction
    - `QT_(NS|EW)`: Partnership quick tricks totals
    - `QT_(NS|EW)_[SHDC]`: Partnership quick tricks by suit

    Notes:
    - Standard QT values: AK=2.0, AQ=1.5, A=1.0, KQ=1.0, K=0.5
    """
    qt_dict = {'AK': 2.0, 'AQ': 1.5, 'A': 1.0, 'KQ': 1.0, 'K': 0.5}
    
    # Calculate QT for each suit
    qt_expr = [
        pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('AK')).then(pl.lit(2.0, dtype=pl.Float32))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('AQ')).then(pl.lit(1.5, dtype=pl.Float32))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('A')).then(pl.lit(1.0, dtype=pl.Float32))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('KQ')).then(pl.lit(1.0, dtype=pl.Float32))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('K')).then(pl.lit(0.5, dtype=pl.Float32))
        .otherwise(pl.lit(0.0, dtype=pl.Float32)).alias(f'QT_{d}_{s}')
        for d in 'NESW' for s in 'SHDC'
    ]
    
    # Apply suit QT calculations
    df = df.with_columns(qt_expr)
    
    # Calculate QT for each direction
    direction_qt = [
        pl.sum_horizontal([pl.col(f'QT_{d}_{s}') for s in 'SHDC']).alias(f'QT_{d}')
        for d in 'NESW'
    ]
    
    # Apply direction QT calculations
    df = df.with_columns(direction_qt)
    
    # Calculate partnership QT
    partnership_qt = [
        (pl.col('QT_N') + pl.col('QT_S')).alias('QT_NS'),
        (pl.col('QT_E') + pl.col('QT_W')).alias('QT_EW')
    ]
    
    # Apply partnership QT calculations
    df = df.with_columns(partnership_qt)
    
    # Calculate partnership QT by suit
    partnership_qt_suit = [
        (pl.col(f'QT_N_{s}') + pl.col(f'QT_S_{s}')).alias(f'QT_NS_{s}')
        for s in 'SHDC'
    ] + [
        (pl.col(f'QT_E_{s}') + pl.col(f'QT_W_{s}')).alias(f'QT_EW_{s}')
        for s in 'SHDC'
    ]
    
    # Apply partnership QT by suit calculations
    return df.with_columns(partnership_qt_suit)


def add_quick_losers(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate Quick Losers for bridge hands using vectorized pattern matching.
    
    Purpose:
    - Compute quick losers based on missing honor combinations
    - Generate QL totals by direction, suit, and partnership
    - Provide offensive strength assessment for hand evaluation (Losing Trick Count concept)

    Parameters:
    - df: DataFrame with suit string columns

    Returns:
    - DataFrame with added Quick Losers columns

    Input columns:
    - `Suit_[NESW]_[SHDC]`: Suit strings for pattern matching

    Output columns:
    - `QL_[NESW]_[SHDC]`: Quick losers by direction and suit
    - `QL_[NESW]`: Total quick losers by direction
    - `QL_(NS|EW)`: Partnership quick losers totals
    - `QL_(NS|EW)_[SHDC]`: Partnership quick losers by suit

    Notes:
    - Standard QL values: Void=0, Singleton(A)=0, Singleton(x)=1
    - Doubleton: AK=0, Ax/Kx=1, xx=2
    - 3+ cards: 3 minus count of A, K, Q present (0-3 losers)
    """
    
    # Calculate QL for each suit
    ql_expr = [
        pl.when(pl.col(f'Suit_{d}_{s}').str.len_chars() == 0).then(pl.lit(0, dtype=pl.UInt8))
        .when(pl.col(f'Suit_{d}_{s}').str.len_chars() == 1).then(
            pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('A')).then(pl.lit(0, dtype=pl.UInt8))
            .otherwise(pl.lit(1, dtype=pl.UInt8)))
        .when(pl.col(f'Suit_{d}_{s}').str.len_chars() == 2).then(
            pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('AK')).then(pl.lit(0, dtype=pl.UInt8))
            .when(pl.col(f'Suit_{d}_{s}').str.starts_with('A')).then(pl.lit(1, dtype=pl.UInt8))
            .when(pl.col(f'Suit_{d}_{s}').str.starts_with('K')).then(pl.lit(1, dtype=pl.UInt8))
            .otherwise(pl.lit(2, dtype=pl.UInt8)))
        .otherwise(pl.lit(3, dtype=pl.UInt8)
            - pl.col(f'Suit_{d}_{s}').str.contains('A').cast(pl.UInt8)
            - pl.col(f'Suit_{d}_{s}').str.contains('K').cast(pl.UInt8)
            - pl.col(f'Suit_{d}_{s}').str.contains('Q').cast(pl.UInt8))
        .alias(f'QL_{d}_{s}')
        for d in 'NESW' for s in 'SHDC'
    ]
    
    # Apply suit QL calculations
    df = df.with_columns(ql_expr)
    
    # Calculate QL for each direction
    direction_ql = [
        pl.sum_horizontal([pl.col(f'QL_{d}_{s}') for s in 'SHDC']).alias(f'QL_{d}')
        for d in 'NESW'
    ]
    
    # Apply direction QL calculations
    df = df.with_columns(direction_ql)
    
    # Calculate partnership QL
    partnership_ql = [
        (pl.col('QL_N') + pl.col('QL_S')).alias('QL_NS'),
        (pl.col('QL_E') + pl.col('QL_W')).alias('QL_EW')
    ]
    
    # Apply partnership QL calculations
    df = df.with_columns(partnership_ql)
    
    # Calculate partnership QL by suit
    partnership_ql_suit = [
        (pl.col(f'QL_N_{s}') + pl.col(f'QL_S_{s}')).alias(f'QL_NS_{s}')
        for s in 'SHDC'
    ] + [
        (pl.col(f'QL_E_{s}') + pl.col(f'QL_W_{s}')).alias(f'QL_EW_{s}')
        for s in 'SHDC'
    ]
    
    # Apply partnership QL by suit calculations
    return df.with_columns(partnership_ql_suit)


def add_losing_trick_count(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate Losing Trick Count (LTC) for bridge hands using vectorized pattern matching.
    
    Purpose:
    - Compute LTC for hand evaluation and bidding decisions
    - Generate LTC totals by direction, suit, and partnership
    - Standard bridge hand evaluation method (24 minus combined LTC = expected tricks)

    Parameters:
    - df: DataFrame with suit string columns

    Returns:
    - DataFrame with added Losing Trick Count columns

    Input columns:
    - `Suit_[NESW]_[SHDC]`: Suit strings for pattern matching

    Output columns:
    - `LTC_[NESW]_[SHDC]`: Losing trick count by direction and suit
    - `LTC_[NESW]`: Total LTC by direction
    - `LTC_(NS|EW)`: Partnership LTC totals
    - `LTC_(NS|EW)_[SHDC]`: Partnership LTC by suit

    Notes:
    - Standard LTC values: Void=0, Singleton(A)=0, Singleton(x)=1
    - Doubleton: AK=0, Ax/Kx=1, xx=2
    - 3+ cards: 3 minus count of A, K, Q present (0-3 losers per suit)
    - Combined tricks estimate: 24 - LTC_NS - LTC_EW
    """
    
    # Calculate LTC for each suit
    ltc_expr = [
        pl.when(pl.col(f'Suit_{d}_{s}').str.len_chars() == 0).then(pl.lit(0, dtype=pl.UInt8))
        .when(pl.col(f'Suit_{d}_{s}').str.len_chars() == 1).then(
            pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('A')).then(pl.lit(0, dtype=pl.UInt8))
            .otherwise(pl.lit(1, dtype=pl.UInt8)))
        .when(pl.col(f'Suit_{d}_{s}').str.len_chars() == 2).then(
            pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('AK')).then(pl.lit(0, dtype=pl.UInt8))
            .when(pl.col(f'Suit_{d}_{s}').str.starts_with('A')).then(pl.lit(1, dtype=pl.UInt8))
            .when(pl.col(f'Suit_{d}_{s}').str.starts_with('K')).then(pl.lit(1, dtype=pl.UInt8))
            .otherwise(pl.lit(2, dtype=pl.UInt8)))
        .otherwise(pl.lit(3, dtype=pl.UInt8)
            - pl.col(f'Suit_{d}_{s}').str.contains('A').cast(pl.UInt8)
            - pl.col(f'Suit_{d}_{s}').str.contains('K').cast(pl.UInt8)
            - pl.col(f'Suit_{d}_{s}').str.contains('Q').cast(pl.UInt8))
        .alias(f'LTC_{d}_{s}')
        for d in 'NESW' for s in 'SHDC'
    ]
    
    # Apply suit LTC calculations
    df = df.with_columns(ltc_expr)
    
    # Calculate LTC for each direction
    direction_ltc = [
        pl.sum_horizontal([pl.col(f'LTC_{d}_{s}') for s in 'SHDC']).alias(f'LTC_{d}')
        for d in 'NESW'
    ]
    
    # Apply direction LTC calculations
    df = df.with_columns(direction_ltc)
    
    # Calculate partnership LTC
    partnership_ltc = [
        (pl.col('LTC_N') + pl.col('LTC_S')).alias('LTC_NS'),
        (pl.col('LTC_E') + pl.col('LTC_W')).alias('LTC_EW')
    ]
    
    # Apply partnership LTC calculations
    df = df.with_columns(partnership_ltc)
    
    # Calculate partnership LTC by suit
    partnership_ltc_suit = [
        (pl.col(f'LTC_N_{s}') + pl.col(f'LTC_S_{s}')).alias(f'LTC_NS_{s}')
        for s in 'SHDC'
    ] + [
        (pl.col(f'LTC_E_{s}') + pl.col(f'LTC_W_{s}')).alias(f'LTC_EW_{s}')
        for s in 'SHDC'
    ]
    
    # Apply partnership LTC by suit calculations
    return df.with_columns(partnership_ltc_suit)


def add_suit_lengths(df: pl.DataFrame) -> pl.DataFrame:
    """Create suit length columns from suit string columns using vectorized operations.

    Purpose:
    - Calculate suit lengths for all directions and suits
    - Foundation for distribution analysis and bidding evaluation
    - Enable suit-based strategic calculations

    Parameters:
    - df: DataFrame containing suit string columns

    Returns:
    - DataFrame with added suit length columns

    Input columns:
    - `Suit_[NESW]_[SHDC]`: Suit strings for length calculation

    Output columns:
    - `SL_[NESW]_[SHDC]`: Suit lengths by direction and suit
    """
    sl_nesw_columns = [
        pl.col(f"Suit_{direction}_{suit}").str.len_chars().alias(f"SL_{direction}_{suit}")
        for direction in "NESW"
        for suit in "SHDC"
    ]
    return df.with_columns(sl_nesw_columns)


def add_pair_suit_lengths(df: pl.DataFrame) -> pl.DataFrame:
    """Create partnership suit length columns from individual direction suit lengths.

    Purpose:
    - Calculate combined suit lengths for partnerships
    - Enable partnership-level distribution analysis
    - Support fit identification and trump suit evaluation

    Parameters:
    - df: DataFrame containing individual direction suit lengths

    Returns:
    - DataFrame with added partnership suit length columns

    Input columns:
    - `SL_[NESW]_[SHDC]`: Individual direction suit lengths

    Output columns:
    - `SL_(NS|EW)_[SHDC]`: Partnership suit lengths
    """
    sl_ns_ew_columns = [
        pl.sum_horizontal(f"SL_{pair[0]}_{suit}", f"SL_{pair[1]}_{suit}").alias(f"SL_{pair}_{suit}")
        for pair in ['NS', 'EW']
        for suit in "SHDC"
    ]
    return df.with_columns(sl_ns_ew_columns)


def add_suit_length_arrays(df: pl.DataFrame, direction: str) -> pl.DataFrame:
    """Create suit length array columns for a specific direction using vectorized operations.

    Purpose:
    - Generate structured suit length representations for pattern analysis
    - Create sorted distributions for hand shape classification
    - Enable advanced distribution-based calculations

    Parameters:
    - df: DataFrame containing suit length columns
    - direction: Single direction (N/E/S/W) to process

    Returns:
    - DataFrame with added suit length array columns

    Input columns:
    - `SL_{direction}_[SHDC]`: Suit lengths for specified direction

    Output columns:
    - `SL_{direction}_CDHS`: List of suit lengths in CDHS order
    - `SL_{direction}_CDHS_SJ`: String-joined suit lengths (e.g., "4-4-3-2")
    - `SL_{direction}_ML`: Suit lengths sorted descending (longest first)
    - `SL_{direction}_ML_I`: Indices of suits in descending length order
    - `SL_{direction}_ML_SJ`: String-joined sorted lengths
    - `SL_{direction}_ML_I_SJ`: String-joined sorted indices
    """
    # Build base list columns
    cdhs_expr = pl.concat_list([
        pl.col(f"SL_{direction}_C"),
        pl.col(f"SL_{direction}_D"),
        pl.col(f"SL_{direction}_H"),
        pl.col(f"SL_{direction}_S"),
    ]).alias(f"SL_{direction}_CDHS")

    cdhs_sj_expr = (
        pl.col(f"SL_{direction}_C").cast(pl.String) + pl.lit("-") +
        pl.col(f"SL_{direction}_D").cast(pl.String) + pl.lit("-") +
        pl.col(f"SL_{direction}_H").cast(pl.String) + pl.lit("-") +
        pl.col(f"SL_{direction}_S").cast(pl.String)
    ).alias(f"SL_{direction}_CDHS_SJ")

    ml_expr = pl.concat_list([
        pl.col(f"SL_{direction}_C"),
        pl.col(f"SL_{direction}_D"),
        pl.col(f"SL_{direction}_H"),
        pl.col(f"SL_{direction}_S"),
    ]).list.sort(descending=True).alias(f"SL_{direction}_ML")

    # Vectorized ML_I (no list.argsort; sort structs by -value then take idx)
    ml_i_expr = (
        pl.concat_list([
            pl.struct([(-pl.col(f"SL_{direction}_C").cast(pl.Int8)).alias("val_neg"), pl.lit(0).alias("idx")]),
            pl.struct([(-pl.col(f"SL_{direction}_D").cast(pl.Int8)).alias("val_neg"), pl.lit(1).alias("idx")]),
            pl.struct([(-pl.col(f"SL_{direction}_H").cast(pl.Int8)).alias("val_neg"), pl.lit(2).alias("idx")]),
            pl.struct([(-pl.col(f"SL_{direction}_S").cast(pl.Int8)).alias("val_neg"), pl.lit(3).alias("idx")]),
        ])
        .list.sort()  # lexicographic on (val_neg, idx)
        .list.eval(pl.element().struct.field("idx"))
        .alias(f"SL_{direction}_ML_I")
    )

    # Materialize base first, then derive SJ strings
    df = df.with_columns([cdhs_expr, cdhs_sj_expr, ml_expr, ml_i_expr])

    ml_sj_expr = (
        pl.col(f"SL_{direction}_ML").list.eval(pl.element().cast(pl.Utf8)).list.join("-")
        .alias(f"SL_{direction}_ML_SJ")
    )
    ml_i_sj_expr = (
        pl.col(f"SL_{direction}_ML_I").list.eval(pl.element().cast(pl.Utf8)).list.join("-")
        .alias(f"SL_{direction}_ML_I_SJ")
    )

    df = df.with_columns([ml_sj_expr, ml_i_sj_expr])
    
    return df


def add_law_of_total_tricks(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate Law of Total Tricks (LoTT) for both partnerships.

    Purpose:
    - Compute combined partnership trump suit lengths
    - Support competitive bidding decisions based on LoTT
    - Provide distribution-based tactical guidance

    Parameters:
    - df: DataFrame containing partnership suit lengths

    Returns:
    - DataFrame with added LoTT columns

    Input columns:
    - `SL_(NS|EW)_[SHDC]`: Partnership suit lengths

    Output columns:
    - `LoTT_NS`: Total suit length for North-South partnership
    - `LoTT_EW`: Total suit length for East-West partnership

    Notes:
    - LoTT suggests total tricks ≈ combined trump length of both partnerships
    """
    df = df.with_columns([
        (pl.col('SL_NS_C') + pl.col('SL_NS_D') + pl.col('SL_NS_H') + pl.col('SL_NS_S')).alias('LoTT_NS'),
        (pl.col('SL_EW_C') + pl.col('SL_EW_D') + pl.col('SL_EW_H') + pl.col('SL_EW_S')).alias('LoTT_EW'),
    ])
    return df


# =============================================================================
# SECTION 3: CACHING & DD/SD INFRASTRUCTURE 
# =============================================================================
# Used by: DD_SD_Augmenter setup and preparation phase
# Purpose: Cache management, score tables, DD/SD computation infrastructure
# Temporal order: Third - these set up the computational infrastructure
# =============================================================================

def update_hand_records_cache(hrs_cache_df: pl.DataFrame, new_df: pl.DataFrame) -> pl.DataFrame:
    """Update hand records cache by merging new data with existing cache.

    Purpose:
    - Efficiently merge new hand record data into existing cache
    - Handle duplicate PBNs by updating existing records or adding new ones
    - Maintain data integrity and optimize for memory usage

    Parameters:
    - hrs_cache_df: Existing hand records cache DataFrame
    - new_df: New hand records data to merge into cache

    Returns:
    - pl.DataFrame: Updated cache with merged data

    Input columns:
    - Both DataFrames should have 'PBN' column and matching schemas

    Output columns:
    - Same as input DataFrames, with updated/added records
    """
    # TODO(polars): Replace Python set operations below with pure-Polars joins (semi/anti)
    # for pbns_to_update/add calculations to avoid materializing large Python sets/lists.
    # Log initial row counts
    logger.info(f"Initial hrs_cache_df rows: {hrs_cache_df.height}")
    logger.info(f"New data rows: {new_df.height}")
    
    # Early return if no new data to process
    if new_df.is_empty():
        logger.info("No new data to process, returning original cache")
        return hrs_cache_df
    
    # Calculate which PBNs will be updated vs added
    existing_pbns = set(hrs_cache_df['PBN'].to_list())
    new_pbns = set(new_df['PBN'].to_list())
    
    pbns_to_update = existing_pbns & new_pbns  # intersection
    pbns_to_add = new_pbns - existing_pbns      # difference
    
    logger.info(f"PBNs to update (existing): {len(pbns_to_update)}")
    logger.info(f"PBNs to add (new): {len(pbns_to_add)}")
    
    # Check for duplicate PBNs in new_df with different Dealer/Vul combinations
    new_df_pbn_counts = new_df['PBN'].value_counts()
    duplicate_pbns_in_new = new_df_pbn_counts.filter(pl.col('count') > 1)
    if duplicate_pbns_in_new.height > 0:
        logger.warning(f"Note: {duplicate_pbns_in_new.height} PBNs in new data have multiple Dealer/Vul combinations")
        logger.warning(f"      This will result in more rows than unique PBNs")
    
    # Calculate expected final row count more accurately
    # The update operation replaces existing rows with matching PBNs
    # Then we add all rows from new_df that don't exist in hrs_cache_df
    expected_final_rows = hrs_cache_df.height - len(pbns_to_update) + new_df.height
    logger.info(f"Expected final rows: {expected_final_rows} (existing: {hrs_cache_df.height} - updated: {len(pbns_to_update)} + new: {len(pbns_to_add)})")

    # check for differing dtypes
    common_cols = set(hrs_cache_df.columns) & set(new_df.columns)
    dtype_diffs = {
        col: (hrs_cache_df[col].dtype, new_df[col].dtype)
        for col in common_cols
        if hrs_cache_df[col].dtype != new_df[col].dtype and new_df[col].dtype != pl.Null
    }
    assert len(dtype_diffs) == 0, f"Differing dtypes: {dtype_diffs}"

    # Update existing rows (only columns from new_df)
    hrs_cache_df = hrs_cache_df.update(new_df, on='PBN')
    
    # Add missing columns to new_df ONLY for new rows
    missing_columns = set(hrs_cache_df.columns) - set(new_df.columns)
    new_rows = new_df.join(hrs_cache_df.select('PBN'), on='PBN', how='anti')
    if new_rows.height > 0:
        new_rows = new_rows.with_columns([
            pl.lit(None).alias(col) for col in missing_columns
        ])
        hrs_cache_df = pl.concat([hrs_cache_df, new_rows.select(hrs_cache_df.columns)])
        logger.info(f"Added {len(missing_columns)} missing columns to {new_rows.height} new rows")
    
    logger.info(f"Final hrs_cache_df rows: {hrs_cache_df.height}")
    logger.info(f"Operation completed successfully")
    
    return hrs_cache_df


# calculate dict of contract result scores. each column contains (non-vul,vul) scores for each trick taken. sets are always penalty doubled.
def precompute_contract_score_tables() -> Tuple[Dict[Tuple, int], Dict[Tuple, int], pl.DataFrame]:
    """Precompute bridge contract scores for all levels/strains/tricks/vulnerability/doubling.

    Purpose:
    - Build fast lookup dicts and a compact DataFrame for score computations.

    Returns:
    - all_scores_d: dict keyed by (level, strain, tricks, vul_is_both: bool, dbl: ""|"X"|"XX") -> score
    - scores_d: dict keyed by (level, strain, tricks, vul_is_both: bool) -> score (assumes penalty= doubled for sets)
    - scores_df: DataFrame with columns `Score_{level}{strain}` as [NV, V] pairs for tricks 0..13
    """
    scores_d = {}
    all_scores_d = {(None,None,None,None,None):0} # PASS

    strain_to_denom = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
    for strain_char in 'SHDCN':
        strain_index = 'CDHSN'.index(strain_char) # [3,2,1,0,4]
        denom = strain_to_denom[strain_index]
        for level in range(1,8): # contract level
            for tricks in range(14):
                result = tricks-6-level
                # sets are always penalty doubled
                scores_d[(level,strain_char,tricks,False)] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(Vul.none)
                scores_d[(level,strain_char,tricks,True)] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(Vul.both)
                # calculate all possible scores
                all_scores_d[(level,strain_char,tricks,False,'')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed,result=result).score(Vul.none)
                all_scores_d[(level,strain_char,tricks,False,'X')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.doubled,result=result).score(Vul.none)
                all_scores_d[(level,strain_char,tricks,False,'XX')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.redoubled,result=result).score(Vul.none)
                all_scores_d[(level,strain_char,tricks,True,'')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed,result=result).score(Vul.both)
                all_scores_d[(level,strain_char,tricks,True,'X')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.doubled,result=result).score(Vul.both)
                all_scores_d[(level,strain_char,tricks,True,'XX')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.redoubled,result=result).score(Vul.both)

    # create score dataframe from dict - vectorized approach
    # Generate all column names and data at once to avoid nested loops
    columns = []
    data = []
    
    for suit in 'SHDCN':
        for level in range(1, 8):
            col_name = f'Score_{level}{suit}'
            columns.append(col_name)
            col_data = [[scores_d[(level,suit,i,False)], scores_d[(level,suit,i,True)]] for i in range(14)]
            data.append(col_data)
    
    # Transpose and create DataFrame efficiently
    scores_df = pl.DataFrame(dict(zip(columns, data)), orient='row')
    return all_scores_d, scores_d, scores_df


def expand_scores_by_vulnerability(scores_df: pl.DataFrame) -> pl.DataFrame:
    """Split each `Score_{level}{strain}` column's [NV,V] lists into separate NV and V columns.

    Purpose:
    - Expand vulnerable/non-vulnerable score arrays into separate columns
    - Enable vulnerability-specific score calculations
    - Create foundation for conditional scoring operations

    Parameters:
    - scores_df: DataFrame from `precompute_contract_score_tables()`

    Returns:
    - DataFrame with expanded score columns

    Input columns:
    - `Score_{level}{strain}`: List columns with [NV, V] score pairs

    Output columns:
    - `Score_{level}{strain}_NV`: Non-vulnerable scores for each contract
    - `Score_{level}{strain}_V`: Vulnerable scores for each contract
    """
    # Pre-compute score columns using optimized approach
    score_columns = {f'Score_{level}{suit}': scores_df[f'Score_{level}{suit}']
                    for level in range(1, 8) for suit in 'CDHSN'}

    # Create a DataFrame from the score_columns dictionary
    df_scores = pl.DataFrame(score_columns)

    # Generate all exploded expressions at once for better performance
    exploded_expressions = [
        expr for col in df_scores.columns
        for expr in [
            pl.col(col).list.get(0).alias(f"{col}_NV"),  # Non-vulnerable score
            pl.col(col).list.get(1).alias(f"{col}_V")    # Vulnerable score
        ]
    ]

    return df_scores.with_columns(exploded_expressions).drop(df_scores.columns)


# =============================================================================
# SECTION 4: DD/SD CORE COMPUTATION FUNCTIONS
# =============================================================================
# Used by: DD_SD_Augmenter during actual computation phase
# Purpose: Core double dummy and single dummy calculations
# Temporal order: Fourth - these perform the heavy computational work
# =============================================================================

def display_dd_deals(deals: list[Deal], dd_result_tables: list[Any], deal_index: int = 0, max_display: int = 4) -> None:
    """Pretty-print a few deals and their double-dummy result tables.

    Purpose:
    - Display bridge deals and their corresponding double-dummy analysis results
    - Provide debugging and inspection capability for deal generation and analysis
    - Output formatted deal information to logs for verification

    Parameters:
    - deals: list of endplay `Deal` objects.
    - dd_result_tables: list of result tables aligned with `deals`.
    - deal_index: starting index to display.
    - max_display: number of deals to show.

    Input columns:
    - None (operates on Deal objects and result tables, not DataFrame columns).

    Output columns:
    - None (logs output, does not create DataFrame columns).

    Returns:
    - None (void function that logs output).
    """
    for dd, rt in zip(deals[deal_index:deal_index+max_display], dd_result_tables[deal_index:deal_index+max_display]):
        deal_index += 1
        logger.info(f"Deal: {deal_index}")
        logger.info(str(dd))
        rt.pprint()


# todo: could save a couple seconds by creating dict of deals
def solve_dd_for_deals(deals: list[Deal], batch_size: int = 40, output_progress: bool = False, progress: Optional[Any] = None) -> list[Any]:
    """Compute double-dummy tables for a list of `Deal`s in batches.

    Purpose:
    - Calculate double-dummy analysis results for multiple bridge deals efficiently
    - Process deals in batches to optimize memory usage and performance
    - Provide progress tracking for long-running computations
    - Interface with endplay library's calc_all_tables function

    Parameters:
    - deals: tuple/list of endplay `Deal`.
    - batch_size: number of deals per `calc_all_tables` call.
    - output_progress: show progress to stdout or provided progress handler.
    - progress: tqdm/streamlit-like object with `set_description`/`progress`.

    Input columns:
    - None (operates on Deal objects, not DataFrame columns).

    Output columns:
    - None (returns result tables, not DataFrame columns).

    Returns:
    - List of result tables, in the same order as deals.
    """
    all_result_tables = []
    for i,b in enumerate(range(0,len(deals),batch_size)):
        if output_progress:
            if i % 100 == 0: # only show progress every 100 batches
                percent_complete = int(b*100/len(deals))
                if progress:
                    if hasattr(progress, 'progress'): # streamlit
                        progress.progress(percent_complete, f"{percent_complete}%: Double dummies calculated for {b} of {len(deals)} unique deals.")
                    elif hasattr(progress, 'set_description'): # tqdm
                        progress.set_description(f"{percent_complete}%: Double dummies calculated for {b} of {len(deals)} unique deals.")
                else:
                    logger.info(f"{percent_complete}%: Double dummies calculated for {b} of {len(deals)} unique deals.")
        result_tables = calc_all_tables(deals[b:b+batch_size])
        all_result_tables.extend(result_tables)
    if output_progress: 
        if progress:
            if hasattr(progress, 'progress'): # streamlit
                progress.progress(100, f"100%: Double dummies calculated for {len(deals)} unique deals.")
                progress.empty() # hmmm, this removes the progress bar so fast that 100% message won't be seen.
            elif hasattr(progress, 'set_description'): # tqdm
                progress.set_description(f"100%: Double dummies calculated for {len(deals)} unique deals.")
        else:
            logger.info(f"100%: Double dummies calculated for {len(deals)} unique deals.")
    return all_result_tables


def compute_dd_trick_tables(hrs_df: pl.DataFrame, hrs_cache_df: pl.DataFrame, max_dd_adds: Optional[int] = None, output_progress: bool = True, progress: Optional[Any] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Compute and return double-dummy trick tables for missing PBNs.

    Purpose:
    - Calculate double-dummy trick counts for bridge deals missing DD analysis
    - Efficiently batch process unique PBNs to avoid redundant calculations
    - Maintain cache of computed results to speed up subsequent operations
    - Generate DD columns showing optimal trick counts for each declarer/strain combination

    Parameters:
    - hrs_df: source hand-records DataFrame (provides PBN universe).
    - hrs_cache_df: cache DataFrame with existing DD columns.
    - max_dd_adds: optional cap on number of PBNs to process.
    - output_progress: whether to print/provide progress.
    - progress: tqdm/streamlit-like progress sink.

    Input columns:
    - `PBN`: Bridge deal notation string (exactly 69 characters).

    Output columns:
    - `DD_[NESW]_[SHDCN]`: Double-dummy trick counts for each declarer/strain (pl.UInt8).

    Returns:
    - dd_df: DataFrame of new rows with `PBN` and `DD_[NESW]_[SHDCN]` columns only.
    - unique_dd_tables_d: dict mapping PBN -> endplay DD table.
    """
    # Calculate double dummy scores only
    logger.info(f"{hrs_df.height=}")
    logger.info(f"{hrs_cache_df.height=}")
    assert hrs_df['PBN'].null_count() == 0, "PBNs in df must be non-null"
    assert hrs_df.filter(pl.col('PBN').str.len_chars().ne(69)).height == 0, hrs_df.filter(pl.col('PBN').str.len_chars().ne(69))
    assert hrs_cache_df['PBN'].null_count() == 0, "PBNs in hrs_cache_df must be non-null"
    assert hrs_cache_df.filter(pl.col('PBN').str.len_chars().ne(69)).height == 0, hrs_cache_df.filter(pl.col('PBN').str.len_chars().ne(69))
    unique_hrs_df_pbns = set(hrs_df['PBN']) # could be non-unique PBN with difference Dealer, Vul.
    logger.info(f"{len(unique_hrs_df_pbns)=}")
    hrs_cache_with_nulls_df = hrs_cache_df.filter(pl.col('DD_N_C').is_null())
    logger.info(f"{len(hrs_cache_with_nulls_df)=}")
    hrs_cache_with_nulls_pbns = hrs_cache_with_nulls_df['PBN']
    logger.info(f"{len(hrs_cache_with_nulls_pbns)=}")
    unique_hrs_cache_with_nulls_pbns = set(hrs_cache_with_nulls_pbns)
    logger.info(f"{len(unique_hrs_cache_with_nulls_pbns)=}")
    
    hrs_cache_all_pbns = set(hrs_cache_df['PBN'])
    pbns_to_add = set(unique_hrs_df_pbns) - hrs_cache_all_pbns  # In hrs_df but NOT in hrs_cache_df
    logger.info(f"{len(pbns_to_add)=}")
    pbns_to_replace = set(unique_hrs_df_pbns).intersection(set(unique_hrs_cache_with_nulls_pbns))  # In both, with nulls in hrs_cache_df
    logger.info(f"{len(pbns_to_replace)=}")
    pbns_to_process = pbns_to_add.union(pbns_to_replace)
    logger.info(f"{len(pbns_to_process)=}")
    
    if max_dd_adds is not None:
        pbns_to_process = list(pbns_to_process)[:max_dd_adds]
        logger.info(f"limit: {max_dd_adds=} {len(pbns_to_process)=}")
    
    cleaned_pbns = [Deal(pbn) for pbn in pbns_to_process]
    assert all([pbn == dpbn.to_pbn() for pbn,dpbn in zip(pbns_to_process,cleaned_pbns)]), [(pbn,dpbn.to_pbn()) for pbn,dpbn in zip(pbns_to_process,cleaned_pbns) if pbn != dpbn.to_pbn()] # usually a sort order issue which should have been fixed in previous step
    unique_dd_tables = solve_dd_for_deals(cleaned_pbns, output_progress=output_progress, progress=progress)
    logger.info(f"{len(unique_dd_tables)=}")
    unique_dd_tables_d = {deal.to_pbn():rt for deal,rt in zip(cleaned_pbns,unique_dd_tables)}
    logger.info(f"{len(unique_dd_tables_d)=}")

    # Create dataframe of double dummy scores only - vectorized approach
    dd_columns = {f'DD_{direction}_{suit}':pl.UInt8 for suit in 'SHDCN' for direction in 'NESW'}
    dd_column_names = list(dd_columns.keys())
    
    # Collect all data at once using list comprehensions
    valid_pbns = [pbn for pbn in pbns_to_process if pbn in unique_dd_tables_d]
    
    if valid_pbns:
        # Create data matrix efficiently
        data_rows = []
        for pbn in valid_pbns:
            dd_rows = sum(unique_dd_tables_d[pbn].to_list(), []) # flatten dd_table
            data_rows.append([pbn] + dd_rows)
        
        # Build dictionary for DataFrame creation
        d = {
            'PBN': [row[0] for row in data_rows],
            **{col: [row[i+1] for row in data_rows] for i, col in enumerate(dd_column_names)}
        }
    else:
        # Handle empty case
        d = {'PBN': [], **{col: [] for col in dd_column_names}}

    # Create a DataFrame using only the keys in dictionary d while maintaining the schema from hrs_cache_df
    filtered_schema = {k: hrs_cache_df[k].dtype for k, v in hrs_cache_df.schema.items() if k in d}
    logger.info(f"filtered_schema: {filtered_schema}")
    error_filtered_schema = {k: None for k, v in d.items() if k not in hrs_cache_df.columns}
    logger.info(f"error_filtered_schema: {error_filtered_schema}")
    assert len(error_filtered_schema) == 0, f"error_filtered_schema: {error_filtered_schema}"
    dd_df = pl.DataFrame(d, schema=filtered_schema)
    return dd_df, unique_dd_tables_d
def compute_par_scores_for_missing(hrs_df: pl.DataFrame, hrs_cache_df: pl.DataFrame, unique_dd_tables_d: Dict[str, Any]) -> pl.DataFrame:
    """Calculate par scores and contracts for PBNs that need them using DD tables.

    Purpose:
    - Calculate optimal par scores and contracts for bridge deals using double-dummy analysis
    - Determine theoretical best results for both sides with perfect play
    - Generate par contract recommendations for deal analysis and validation
    - Process only deals missing par analysis to optimize computation time

    Parameters:
    - hrs_df: source hand-records DataFrame.
    - hrs_cache_df: cache DataFrame (provides schema and existing rows).
    - unique_dd_tables_d: dict PBN -> DD table.

    Input columns:
    - `PBN`: Bridge deal notation string (from hrs_df)
    - `Dealer`: Deal rotation indicator (N/E/S/W)
    - `Vul`: Vulnerability status (None/N_S/E_W/Both)

    Output columns:
    - `ParScore`: Optimal score achievable with perfect play (pl.Int32)
    - `ParNumber`: Number of different optimal contracts available (pl.UInt8)
    - `ParContracts`: String list of optimal contract(s) (pl.String)

    Returns:
    - DataFrame with `PBN`, `Dealer`, `Vul`, `ParScore`, `ParNumber`, and `ParContracts`.
    """
    # TODO(polars): Replace per-row Python dict construction with vectorized
    # assembly (e.g., create a list of rows then `pl.DataFrame(rows, schema=schema)`)
    # and use joins to attach Dealer/Vul; avoid Python loops where possible.
    # Calculate par scores using existing double dummy tables
    logger.info(f"{hrs_df.height=}")
    logger.info(f"{hrs_cache_df.height=}")
    logger.info(f"Calculating par scores for {len(unique_dd_tables_d)} deals")
    
    # Assert that there are no None values in 'PBN' in hrs_cache_df
    assert hrs_cache_df['PBN'].null_count() == 0, f"Found {hrs_cache_df['PBN'].null_count()} None values in 'PBN' column of hrs_cache_df"
    
    # untested but interesting
    # # If no new DD tables were calculated, build DD tables from existing cache data
    # if len(unique_dd_tables_d) == 0:
    #     logger.info("No new DD tables provided, building from existing cache data")
    #     # Get PBNs that have DD data but missing ParScore
    #     pbns_with_dd = hrs_cache_df.filter(
    #         pl.col('DD_N_C').is_not_null() & 
    #         pl.col('ParScore').is_null()
    #     )['PBN'].unique().to_list()
        
    #     if len(pbns_with_dd) > 0:
    #         logger.info(f"Found {len(pbns_with_dd)} PBNs with DD data but missing ParScore")
    #         # Build DD tables from existing cache data
    #         from endplay import Deal
    #         unique_dd_tables_d = {}
    #         for pbn in pbns_with_dd:
    #             # Get DD values from cache for this PBN
    #             dd_df = hrs_cache_df.filter(pl.col('PBN') == pbn).select([
    #                 'DD_N_S', 'DD_N_H', 'DD_N_D', 'DD_N_C', 'DD_N_N',
    #                 'DD_E_S', 'DD_E_H', 'DD_E_D', 'DD_E_C', 'DD_E_N',
    #                 'DD_S_S', 'DD_S_H', 'DD_S_D', 'DD_S_C', 'DD_S_N',
    #                 'DD_W_S', 'DD_W_H', 'DD_W_D', 'DD_W_C', 'DD_W_N'
    #             ])
                
    #             if dd_df.height > 0:
    #                 dd_row = dd_df.row(0)
                    
    #                 # Convert to DD table format (4x5 matrix: NESW x SHDCN)
    #                 dd_table = []
    #                 for direction in ['N', 'E', 'S', 'W']:
    #                     row = []
    #                     for suit in ['S', 'H', 'D', 'C', 'N']:
    #                         col_name = f'DD_{direction}_{suit}'
    #                         # Find the index of this column in the selected columns
    #                         col_idx = dd_df.columns.index(col_name)
    #                         row.append(dd_row[col_idx])
    #                     dd_table.append(row)
                    
    #                 unique_dd_tables_d[pbn] = dd_table
    #         logger.info(f"Built DD tables for {len(unique_dd_tables_d)} PBNs from cache")
    #     else:
    #         logger.info("No PBNs found with DD data but missing ParScore")
    
    # Find PBNs that need par score calculation (only those that have DD calculations)
    unique_hrs_df_pbns = set(hrs_df['PBN'])
    logger.info(f"{len(unique_hrs_df_pbns)=}")
    hrs_cache_with_nulls_df = hrs_cache_df.filter(pl.col('ParScore').is_null())
    logger.info(f"{len(hrs_cache_with_nulls_df)=}")
    hrs_cache_with_nulls_pbns = set(hrs_cache_with_nulls_df['PBN'])
    logger.info(f"{len(hrs_cache_with_nulls_pbns)=}")
    
    hrs_cache_all_pbns = set(hrs_cache_df['PBN'])
    pbns_to_add = set(unique_hrs_df_pbns) - hrs_cache_all_pbns
    logger.info(f"{len(pbns_to_add)=}")
    pbns_to_replace = set(unique_hrs_df_pbns).intersection(hrs_cache_with_nulls_pbns)
    logger.info(f"{len(pbns_to_replace)=}")
    pbns_to_process = pbns_to_add.union(pbns_to_replace)
    logger.info(f"{len(pbns_to_process)=}")
    
    # Check which PBNs missing ParScore have DD tables available
    pbns_missing_par = set(hrs_cache_with_nulls_pbns)  # Use the already calculated set
    pbns_with_dd_tables = set(unique_dd_tables_d.keys())
    pbns_missing_par_and_dd = pbns_missing_par - pbns_with_dd_tables
    pbns_missing_par_with_dd = pbns_missing_par & pbns_with_dd_tables
    
    logger.info(f"PBNs missing ParScore: {len(pbns_missing_par)}")
    logger.info(f"PBNs with DD tables available: {len(pbns_with_dd_tables)}")
    logger.info(f"PBNs missing ParScore with DD tables: {len(pbns_missing_par_with_dd)}")
    logger.info(f"PBNs missing ParScore without DD tables: {len(pbns_missing_par_and_dd)}")
    
    # Only process PBNs that have DD tables available
    pbns_to_process = pbns_to_process & pbns_with_dd_tables
    logger.info(f"PBNs to process (with DD tables): {len(pbns_to_process)}")
    
    # Get Dealer/Vul from appropriate source for each PBN type - optimized with Polars operations
    source_dfs = []
    
    if pbns_to_add:
        source_dfs.append(hrs_df.filter(pl.col('PBN').is_in(list(pbns_to_add)))[['PBN','Dealer','Vul']].unique())
    
    if pbns_to_replace:
        # Get rows from hrs_cache_df, but for None values in Dealer/Vul, get them from hrs_df
        cache_rows = hrs_cache_df.filter(pl.col('PBN').is_in(list(pbns_to_replace)))[['PBN','Dealer','Vul']].unique()
        
        # Find rows with None values in Dealer or Vul
        rows_with_none = cache_rows.filter(pl.col('Dealer').is_null() | pl.col('Vul').is_null())
        rows_without_none = cache_rows.filter(pl.col('Dealer').is_not_null() & pl.col('Vul').is_not_null())
        
        # Get Dealer/Vul values from hrs_df for rows with None values
        if rows_with_none.height > 0:
            logger.info(f"Found {rows_with_none.height} rows with None Dealer/Vul values, getting from hrs_df")
            pbn_list = rows_with_none['PBN'].to_list()
            hrs_df_rows = hrs_df.filter(pl.col('PBN').is_in(pbn_list))[['PBN','Dealer','Vul']].unique()
            source_dfs.append(hrs_df_rows)
        
        # Add rows that already have valid Dealer/Vul values
        if rows_without_none.height > 0:
            source_dfs.append(rows_without_none)
    
    # Combine all DataFrames efficiently and convert to rows only at the end
    if source_dfs:
        combined_df = pl.concat(source_dfs)
        source_rows = combined_df.rows()
    else:
        source_rows = []
    
    # Create dataframe of par scores
    d = defaultdict(list)
    for pbn, dealer, vul in source_rows:
        if pbn not in unique_dd_tables_d:
            continue
        d['PBN'].append(pbn)
        d['Dealer'].append(dealer)
        d['Vul'].append(vul)
        parlist = par(unique_dd_tables_d[pbn], VulToEndplayVul_d[vul], DealerToEndPlayDealer_d[dealer])
        d['ParScore'].append(parlist.score)
        d['ParNumber'].append(parlist._data.number)
        contracts = [{
            "Level": str(contract.level),
            "Strain": 'SHDCN'[int(contract.denom)],
            "Doubled": contract.penalty.abbr,
            "Pair_Direction": 'NS' if contract.declarer.abbr in 'NS' else 'EW',
            "Result": contract.result
        } for contract in parlist]
        d['ParContracts'].append(contracts)

    # Create a DataFrame using only the keys in dictionary d while maintaining the schema from hrs_cache_df
    filtered_schema = {k: hrs_cache_df[k].dtype for k, v in hrs_cache_df.schema.items() if k in d}
    logger.info(f"filtered_schema: {filtered_schema}")
    error_filtered_schema = {k: None for k, v in d.items() if k not in hrs_cache_df.columns}
    logger.info(f"error_filtered_schema: {error_filtered_schema}")
    assert len(error_filtered_schema) == 0, f"error_filtered_schema: {error_filtered_schema}"
    par_df = pl.DataFrame(d, schema=filtered_schema)
    logger.info(f"par_df height: {par_df.height}")
    return par_df


def deal_generation_constraints(deal: Deal) -> bool:
    """Dealer constraints for use with `generate_deals`. Currently accepts all deals.

    Purpose:
    - Define constraints for bridge deal generation
    - Currently implemented as pass-through (accepts all deals)
    - Can be extended for specific deal filtering requirements

    Parameters:
    - deal: Deal object representing a bridge hand distribution

    Returns:
    - bool: True if deal meets constraints (currently always True)

    Input columns:
    - None (operates on Deal objects)

    Output columns:
    - None (returns boolean constraint result)
    """
    return True


def generate_single_dummy_deals(predeal_string: str, produce: int, env: Dict[str, Any] = dict(), max_attempts: int = 1000000, seed: int = 42, show_progress: bool = True, strict: bool = True, swapping: int = 0) -> Tuple[Tuple[Deal, ...], list[Any]]:
    """Generate deals for single-dummy simulation given a predeal string.

    Purpose:
    - Generate multiple bridge deals for single-dummy analysis with partial predeals
    - Create deal variations where some seats are fixed and others are randomly dealt
    - Support Monte Carlo simulation for single-dummy trick distribution analysis
    - Provide controlled randomization for bridge deal generation and testing

    Parameters:
    - predeal_string: PBN-like string with two seats elided using '...'.
    - produce: number of deals to generate.
    - env, max_attempts, seed, show_progress, strict, swapping: passed to endplay.dealer.generate_deals.

    Input columns:
    - None (operates on predeal string input, not DataFrame columns).

    Output columns:
    - None (returns Deal objects and result tables, not DataFrame columns).

    Returns:
    - (deals, dd_result_tables) where `deals` is a tuple of Deal and the tables come from compute_dd_deals.
    """
    predeal = Deal(predeal_string)

    deals_t = generate_deals(
        deal_generation_constraints,
        predeal=predeal,
        swapping=swapping,
        show_progress=show_progress,
        produce=produce,
        seed=seed,
        max_attempts=max_attempts,
        env=env,
        strict=strict
        )

    deals = tuple(deals_t) # create a tuple before interop memory goes wonky
    
    return deals, solve_dd_for_deals(deals)


def estimate_sd_trick_distributions(deal: str, produce: int = 100) -> Tuple[Dict[str, pl.DataFrame], Tuple[int, Dict[Tuple[str, str, str], list[float]]]]:
    """Estimate single-dummy trick distributions by side (NS/EW) for a given deal string.

    Purpose:
    - Generate probabilistic trick distributions for single-dummy play analysis
    - Simulate multiple deal variations with one partnership's cards hidden
    - Calculate expected trick counts across different contracts and declarers
    - Support single-dummy analysis for bidding and play evaluation

    Parameters:
    - deal: PBN string for a full deal.
    - produce: number of generated deals per side.

    Input columns:
    - None (operates on PBN deal string, not DataFrame columns).

    Output columns:
    - None (returns analysis dictionaries and DataFrames, not DataFrame columns).

    Returns:
    - SD_Tricks_df: dict with 'NS'/'EW' -> DataFrame of simulated DD tricks.
    - (produce, ns_ew_rows): summary counts as probabilities for each (side, declarer, strain).
    """
    # TODO(polars): Replace per-column value_counts loop with a single melt ->
    # group_by -> normalize -> complete 0..13 -> pivot pipeline; this will
    # dramatically reduce Python overhead and use Polars vectorization.
    # todo: has this been obsoleted by endplay's calc_all_tables 2nd parameter?
    ns_ew_rows = {}
    SD_Tricks_df = {}
    for ns_ew in ['NS','EW']:
        s = deal[2:].split()
        if ns_ew == 'NS':
            s[1] = '...'
            s[3] = '...'
        else:
            s[0] = '...'
            s[2] = '...'
        predeal_string = 'N:'+' '.join(s)
        #logger.info(f"predeal:{predeal_string}")

        sd_deals, sd_dd_result_tables = generate_single_dummy_deals(predeal_string, produce, show_progress=False)

        #show_dd_deals(sd_deals, sd_dd_result_tables, 0, 4)
        # Create DataFrame more efficiently - collect all data at once
        schema = {'SD_Deal': pl.String}
        schema.update({f'SD_Tricks_{d}_{s}': pl.UInt8 for s in 'SHDCN' for d in 'NESW'})
        
        # Build data rows efficiently 
        data_rows = []
        for sddeal, t in zip(sd_deals, sd_dd_result_tables):
            row = [sddeal.to_pbn()]
            row.extend([s for d in t.to_list() for s in d])
            data_rows.append(row)
        
        SD_Tricks_df[ns_ew] = pl.DataFrame(data_rows, schema=schema, orient='row')

        # Vectorized probability calculation for all direction-suit combinations
        for d in 'NESW':
            for s in 'SHDCN':
                col_name = f'SD_Tricks_{d}_{s}'
                # Get value counts as dict and fill missing values efficiently
                vc_result = SD_Tricks_df[ns_ew][col_name].value_counts(normalize=True)
                vc = dict(vc_result.rows())
                
                # Create probability array with proper indexing
                prob_array = [vc.get(i, 0.0) for i in range(14)]
                ns_ew_rows[(ns_ew,d,s)] = prob_array

    return SD_Tricks_df, (produce, ns_ew_rows)


# def append_single_dummy_results(pbns,sd_cache_d,produce=100):
#     for pbn in pbns:
#         if pbn not in sd_cache_d:
#             sd_cache_d[pbn] = estimate_sd_trick_distributions(pbn, produce) # all combinations of declarer pair direction, declarer direction, suit, tricks taken
#     return sd_cache_d


def estimate_sd_trick_distributions_for_df(df: pl.DataFrame, hrs_cache_df: pl.DataFrame, sd_productions: int = 100, max_sd_adds=100, progress: Optional[Any] = None) -> Tuple[Dict[str, pl.DataFrame], pl.DataFrame]:
    """Compute and cache single-dummy probabilities for PBNs needing them.

    Purpose:
    - Generate single-dummy trick distribution probabilities for bridge deals at scale
    - Process only deals missing probability analysis to optimize computation time
    - Create comprehensive probability matrices for all contract combinations
    - Cache results to avoid redundant calculations in future operations

    Parameters:
    - df: source hand-records DataFrame.
    - hrs_cache_df: cache DataFrame with schema including Probs_* columns.
    - sd_productions: number of generated deals per side.
    - max_sd_adds: cap on number of PBNs processed.
    - progress: optional progress sink.

    Input columns:
    - `PBN`: Bridge deal notation string for analysis

    Output columns:
    - `Probs_{NS|EW}_{declarer}_{strain}_{tricks}`: Probability of taking exactly {tricks} tricks (pl.Float32)

    Returns:
    - (sd_dfs_d, hrs_cache_df_updated): dict of side->DataFrame, and updated cache df.
    """
    # TODO(polars): Use anti/semi joins to derive pbns_to_add/replace as DataFrames,
    # and keep them in Polars throughout; avoid Python sets/lists to reduce memory
    # pressure and improve speed on very large datasets.
    # calculate single dummy probabilities. if already calculated use cache value else update e with new result.
    # todo: reduce Python set/dict usage; prefer Polars joins/group_bys for pbns_to_add/replace and probability assembly.
    sd_d = {}
    sd_dfs_d = {}
    assert hrs_cache_df.height == hrs_cache_df.unique(subset=['PBN', 'Dealer', 'Vul']).height, "PBN+Dealer+Vul combinations in hrs_cache_df must be unique"
    
    # Calculate which PBNs to add vs replace
    # Note: Single dummy calculations are the same for a given PBN regardless of Dealer/Vul
    # So we work at the PBN level, but need to handle multiple Dealer/Vul combinations properly
    
    # Find PBNs to add (in df but not in hrs_cache_df)
    df_pbns = set(df['PBN'].to_list())
    hrs_cache_pbns = set(hrs_cache_df['PBN'].to_list())
    pbns_to_add = df_pbns - hrs_cache_pbns
    logger.info(f"PBNs to add: {len(pbns_to_add)}")
    
    # Find PBNs to replace (in both, but with null Probs_Trials in hrs_cache_df)
    # todo: this step takes 3m. find a faster way. perhaps using join?
    pbns_to_replace = set(hrs_cache_df.filter(
        pl.col('PBN').is_in(df['PBN'].to_list()) & 
        pl.col('Probs_Trials').is_null()
    )['PBN'].to_list())
    logger.info(f"PBNs to replace: {len(pbns_to_replace)}")
    
    # Combine all PBNs that need processing
    pbns_to_process = list(pbns_to_add.union(pbns_to_replace))
    logger.info(f"Total unique PBNs to process: {len(pbns_to_process)}")
    if max_sd_adds is not None:
        pbns_to_process = list(pbns_to_process)[:max_sd_adds]
        logger.info(f"limit: {max_sd_adds=} {len(pbns_to_process)=}")
    cleaned_pbns = [Deal(pbn) for pbn in pbns_to_process]
    assert all([pbn == dpbn.to_pbn() for pbn,dpbn in zip(pbns_to_process,cleaned_pbns)]), [(pbn,dpbn.to_pbn()) for pbn,dpbn in zip(pbns_to_process,cleaned_pbns) if pbn != dpbn.to_pbn()] # usually a sort order issue which should have been fixed in previous step
    estimated_processing_time_per_hour = 8000
    logger.info(f"processing time assuming {estimated_processing_time_per_hour}/hour:{len(pbns_to_process)/estimated_processing_time_per_hour} hours")
    for i,pbn in enumerate(pbns_to_process):
        if progress:
            percent_complete = int(i*100/len(pbns_to_process))
            if hasattr(progress, 'progress'): # streamlit
                progress.progress(percent_complete, f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal. This step takes 30 seconds...")
            elif hasattr(progress, 'set_description'): # tqdm
                progress.set_description(f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal. This step takes 30 seconds...")
        else:
            if i < 10 or i % estimated_processing_time_per_hour == 0:
                percent_complete = int(i*100/len(pbns_to_process))
                logger.info(f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
        if not progress and (i < 10 or i % estimated_processing_time_per_hour == 0):
            t = time.time()
        sd_dfs_d[pbn], sd_d[pbn] = estimate_sd_trick_distributions(pbn, sd_productions) # all combinations of declarer pair direction, declarer direction, suit, tricks taken
        if not progress and (i < 10 or i % estimated_processing_time_per_hour == 0):
            logger.info(f"compute_sd_probabilities: time:{time.time()-t} seconds")
        #error
    if progress:
        if hasattr(progress, 'progress'): # streamlit
            progress.progress(100, f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
        elif hasattr(progress, 'set_description'): # tqdm
            progress.set_description(f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
    else:
        logger.info(f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")

    # create single dummy trick taking probability distribution columns - vectorized approach
    if sd_d:
        # Extract PBNs and productions efficiently
        pbns = list(sd_d.keys())
        productions_list = [sd_d[pbn][0] for pbn in pbns]
                
        # Build data dictionary efficiently
        sd_probs_d = {'PBN': pbns, 'Probs_Trials': productions_list}
        
        # Initialize all probability columns with empty lists
        for col_name in PROB_COLUMNS:
            sd_probs_d[col_name] = []
        
        # Fill probability data efficiently
        for pbn in pbns:
            _, probs_d = sd_d[pbn]
            # Create a lookup dict for this PBN's probabilities
            pbn_probs = {}
            for (pair_direction, declarer_direction, suit), probs in probs_d.items():
                for i, t in enumerate(probs):
                    col_name = f'Probs_{pair_direction}_{declarer_direction}_{suit}_{i}'
                    pbn_probs[col_name] = t
            
            # Append values for all columns (0.0 for missing ones)
            for col_name in PROB_COLUMNS:
                sd_probs_d[col_name].append(pbn_probs.get(col_name, 0.0))
    else:
        # Handle empty case
        sd_probs_d = {'PBN': [], 'Probs_Trials': []}
        
    sd_probs_df = pl.DataFrame(sd_probs_d, orient='row')
    sd_probs_df = sd_probs_df.with_columns(pl.col(pl.Float64).cast(pl.Float32))

    # update sd_df with sd_probs_df # doesn't work because sd_df isn't updated unless returned.
    # if sd_df.is_empty():
    #     sd_df = sd_probs_df
    # else:
    #     assert set(sd_df['PBN']).isdisjoint(set(sd_probs_df['PBN']))
    #     assert set(sd_df.columns) == (set(sd_probs_df.columns))
    #     sd_df = pl.concat([sd_df, sd_probs_df.select(sd_df.columns)]) # must reorder columns to match sd_df

    if progress and hasattr(progress, 'empty'):
        progress.empty()

    return sd_dfs_d, sd_probs_df





# def get_cached_sd_data(pbn: str, hrs_d: Dict[str, Any]) -> Dict[str, Union[str, float]]:
#     sd_data = hrs_d[pbn]['SD'][1]
#     row_data = {'PBN': pbn}
#     for (pair_direction, declarer_direction, strain), probs in sd_data.items():
#         col_prefix = f"{pair_direction}_{declarer_direction}_{strain}"
#         for i, prob in enumerate(probs):
#             row_data[f"{col_prefix}_{i}"] = prob
#     return row_data


def add_single_dummy_expected_values(df: pl.DataFrame, scores_df: pl.DataFrame) -> pl.DataFrame:
    """Compute expected values for all contract combinations using single dummy probabilities.

    Purpose:
    - Calculate expected scores for all possible contracts based on trick-taking probabilities
    - Generate comprehensive EV matrix for contract evaluation and analysis
    - Support both vulnerable and non-vulnerable scenarios

    Parameters:
    - df: DataFrame containing hand records with probability data
    - scores_df: DataFrame with scoring information for all contract combinations

    Returns:
    - pl.DataFrame: Input DataFrame with added expected value columns

    Input columns:
    - 'PBN': Position-based notation for the deal
    - 'Probs_*': Single dummy probability columns for trick taking
    - Various vulnerability and deal context columns

    Output columns:
    - 'EV_{pair}_{declarer}_{strain}_{level}': Expected value for each contract combination
    - Where pair ∈ {NS,EW}, declarer ∈ {N,E,S,W}, strain ∈ {S,H,D,C,N}, level ∈ {1-7}
    """

    # retrieve probabilities from cache
    #sd_probs = [get_cached_sd_data(pbn, hrs_d) for pbn in df['PBN']]

    # Create a DataFrame from the extracted sd probs (frequency distribution of tricks).
    #sd_df = pl.DataFrame(sd_probs)

    # todo: look for other places where this is called. duplicated code?
    scores_df_vuls = expand_scores_by_vulnerability(scores_df)

    # takes 2m for 4m rows,5m for 7m rows
    # Define the combinations
    # todo: make global function
    pair_directions = ['NS', 'EW']
    declarer_directions = 'NESW'
    strains = 'SHDCN'
    levels = range(1,8)
    tricks = range(14)
    vuls = ['NV','V']

    # Perform the multiplication
    df = df.with_columns([
        pl.col(f'Probs_{pair_direction}_{declarer_direction}_{strain}_{taken}').mul(score).alias(f'EV_{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}_{taken}_{score}')
        for pair_direction in pair_directions
        for declarer_direction in pair_direction #declarer_directions
        for strain in strains
        for level in levels
        for vul in vuls
        for taken, score in zip(tricks, scores_df_vuls[f'Score_{level}{strain}_{vul}'])
    ])
    #logger.info("Results with prob*score:")
    #display(result)

    # Add a column for the sum (expected value)
    df = df.with_columns([
        pl.sum_horizontal(pl.col(f'^EV_{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}_\\d+_.*$')).alias(f'EV_{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}')
        for pair_direction in pair_directions
        for declarer_direction in pair_direction #declarer_directions
        for strain in strains
        for level in levels
        for vul in vuls
    ])

    #logger.info("\nResults with expected value:")
    return df


def compute_dd_par_and_sd_features(
    df: pl.DataFrame,
    hrs_cache_df: pl.DataFrame,
    sd_productions: int,
    max_dd_adds: Optional[int],
    max_sd_adds: Optional[int],
    output_progress: Optional[bool],
    progress: Optional[Any],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Compute DD, Par, SD features and merge into the working frame and cache.

    Purpose:
    - Orchestrate calculation of double-dummy tables, Par scores, and single-dummy probabilities,
      update the cache, join computed columns back to `df`, and derive EV columns and best contracts.

    Parameters:
    - df: input hand-record DataFrame containing at least `PBN`, `Dealer`, `Vul`.
    - hrs_cache_df: cache DataFrame with schema for DD/Par/SD columns.
    - sd_productions: number of generated deals for single-dummy simulations per side.
    - max_dd_adds: optional cap on number of PBNs to compute DD for.
    - max_sd_adds: optional cap on number of PBNs to compute SD for.
    - output_progress: whether to print/provide progress updates.
    - progress: optional progress sink (tqdm/streamlit-like).

    Input columns:
    - `PBN`, `Dealer`, `Vul` in both `df` and `hrs_cache_df` (for joining).

    Output columns joined into df (from cache computations):
    - `DD_[NESW]_[SHDCN]`, `ParScore`, `ParNumber`, `ParContracts`, and `Probs_*` columns.
    - Derived: `DD_(NS|EW)_[SHDCN]`, EV_* expected values, and best-contract EV maxima.

    Returns:
    - (df_out, hrs_cache_df_out): updated DataFrame and cache DataFrame.
    """
    t0 = time.time()
    all_scores_d, scores_d, scores_df = precompute_contract_score_tables()

    # Double-dummy tables and scores
    dd_df, unique_dd_tables_d = compute_dd_trick_tables(
        df, hrs_cache_df, max_dd_adds, output_progress, progress
    )
    if not dd_df.is_empty():
        hrs_cache_df = update_hand_records_cache(hrs_cache_df, dd_df)

    # Par scores from DD
    par_df = compute_par_scores_for_missing(df, hrs_cache_df, unique_dd_tables_d)
    if not par_df.is_empty():
        hrs_cache_df = update_hand_records_cache(hrs_cache_df, par_df)

    # Single-dummy probabilities
    sd_dfs_d, sd_df = estimate_sd_trick_distributions_for_df(
        df, hrs_cache_df, sd_productions, max_sd_adds, progress
    )
    if not sd_df.is_empty():
        hrs_cache_df = update_hand_records_cache(hrs_cache_df, sd_df)

    # Join back to df
    out_df = df.join(hrs_cache_df, on=['PBN', 'Dealer', 'Vul'], how='inner')

    # Create DD pair strain maxima
    out_df = out_df.with_columns(
        pl.max_horizontal(f"DD_{pair[0]}_{strain}", f"DD_{pair[1]}_{strain}").alias(f"DD_{pair}_{strain}")
        for pair in ['NS', 'EW'] for strain in 'SHDCN'
    )

    # Expected values using SD frequencies and score tables
    out_df = add_single_dummy_expected_values(out_df, scores_df)

    # Best contract EV summaries
    best_contracts_df = identify_best_contracts_by_ev(out_df)
    assert out_df.height == best_contracts_df.height
    out_df = pl.concat([out_df, best_contracts_df], how='horizontal')

    logger.info(f"compute_dd_par_sd_features: time:{time.time()-t0} seconds")
    return out_df, hrs_cache_df


# Function to create columns of max values from various regexes of columns. also creates columns of the column names of the max value.
def find_max_horizontal_value(df: pl.DataFrame, pattern: str) -> Tuple[pl.Expr, pl.Expr]:
    """Return max value and the column name of the max over columns matching a pattern.

    Purpose:
    - Find maximum value across multiple columns matching a regex pattern
    - Identify which column contains the maximum value for each row
    - Support horizontal aggregations and best-value analysis
    - Enable pattern-based column selection for dynamic max operations

    Parameters:
    - df: DataFrame to search.
    - pattern: regex to select columns.

    Input columns:
    - All columns matching the provided regex pattern.

    Output columns:
    - None (returns Polars expressions, not DataFrame columns).

    Returns:
    - (max_expr, col_expr): expressions yielding the row-wise max and the corresponding column name.
    """
    cols = df.select(pl.col(pattern)).columns
    max_expr = pl.max_horizontal(pl.col(pattern))
    col_expr = pl.when(pl.col(cols[0]) == max_expr).then(pl.lit(cols[0]))
    for col in cols[1:]:
        col_expr = col_expr.when(pl.col(col) == max_expr).then(pl.lit(col))
    return max_expr, col_expr.otherwise(pl.lit(""))
# =============================================================================
# SECTION 5: CONTRACT ANALYSIS FUNCTIONS
# =============================================================================
# Used by: FinalContractAugmenter
# Purpose: Contract parsing, analysis, result calculations, best contract identification  
# Temporal order: Fifth - these analyze actual played contracts and outcomes
# =============================================================================

# calculate EV max scores for various regexes including all vulnerabilities. also create columns of the column names of the max values.
def identify_best_contracts_by_ev(df: pl.DataFrame) -> pl.DataFrame:
    """Identify the highest expected value contracts for each pair and vulnerability.

    Purpose:
    - Find optimal contracts based on expected value calculations
    - Generate best contract recommendations for both pairs and vulnerabilities
    - Create column references for maximum EV lookups

    Parameters:
    - df: DataFrame containing expected value columns for all contracts

    Returns:
    - pl.DataFrame: Input DataFrame with added best contract identification columns

    Input columns:
    - 'EV_{pair}_{declarer}_{strain}_{level}': Expected value columns for all contracts
    - Various vulnerability and context columns

    Output columns:
    - 'EV_Max_{pair}_{vul}': Maximum expected value for each pair/vulnerability combination
    - 'EV_Max_Col_{pair}_{vul}': Column name of the contract with maximum expected value
    - Where pair ∈ {NS,EW}, vul ∈ {NV,V}
    """

    # Define the combinations
    pair_directions = ['NS', 'EW']
    declarer_directions = 'NESW'
    strains = 'SHDCN'
    vulnerabilities = ['NV', 'V']

    # Dictionary to store expressions with their aliases as keys
    max_ev_dict = {}

    # all EV columns are already calculated. just need to get the max.

    # Single loop handles all EV Max, Max_Col combinations
    for v in vulnerabilities:
        # Level 4: Overall Max EV for each vulnerability
        ev_columns = f'^EV_(NS|EW)_[NESW]_[SHDCN]_[1-7]_{v}$'
        max_expr, col_expr = find_max_horizontal_value(df, ev_columns)
        max_ev_dict[f'EV_{v}_Max'] = max_expr
        max_ev_dict[f'EV_{v}_Max_Col'] = col_expr

        for pd in pair_directions:
            # Level 3: Max EV for each pair direction and vulnerability
            ev_columns = f'^EV_{pd}_[NESW]_[SHDCN]_[1-7]_{v}$'
            max_expr, col_expr = find_max_horizontal_value(df, ev_columns)
            max_ev_dict[f'EV_{pd}_{v}_Max'] = max_expr
            max_ev_dict[f'EV_{pd}_{v}_Max_Col'] = col_expr

            for dd in pd: #declarer_directions:
                # Level 2: Max EV for each pair direction, declarer direction, and vulnerability
                ev_columns = f'^EV_{pd}_{dd}_[SHDCN]_[1-7]_{v}$'
                max_expr, col_expr = find_max_horizontal_value(df, ev_columns)
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max'] = max_expr
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max_Col'] = col_expr

                for s in strains:
                    # Level 1: Max EV for each combination
                    ev_columns = f'^EV_{pd}_{dd}_{s}_[1-7]_{v}$'
                    max_expr, col_expr = find_max_horizontal_value(df, ev_columns)
                    max_ev_dict[f'EV_{pd}_{dd}_{s}_{v}_Max'] = max_expr
                    max_ev_dict[f'EV_{pd}_{dd}_{s}_{v}_Max_Col'] = col_expr

    # Create expressions list from dictionary
    t = time.time()
    all_max_ev_expr = [expr.alias(alias) for alias, expr in max_ev_dict.items()]
    logger.info(f"identify_best_contracts: all_max_ev_expr created: time:{time.time()-t} seconds")

    # Create a new DataFrame with only the new columns
    # todo: this step is inexplicably slow. appears to take 6 seconds regardless of row count?
    t = time.time()
    df = df.select(all_max_ev_expr)
    logger.info(f"identify_best_contracts: sd_ev_max_df created: time:{time.time()-t} seconds")

    return df


def standardize_contract_format(df: pl.DataFrame) -> pl.Series:
    """Normalize contract format by converting symbols to standard notation.

    Purpose:
    - Standardize contract notation across different input formats
    - Convert suit symbols (♠♥♦♣) to letters (SHDC)
    - Convert NT to N for no-trump contracts

    Parameters:
    - df: DataFrame containing 'Contract' column with various contract formats

    Input columns:
    - 'Contract': Contract string with potential symbol variations

    Output columns:
    - Returns normalized contract series (not added to DataFrame)

    Returns:
    - pl.Series: Normalized contract strings in standard format
    """
    # todo: use strain to strai dict instead of suit symbol. Replace replace() with replace_strict().
    # todo: implement in case 'Contract' is not in self.df.columns but BidLvl, BidSuit, Dbl, Declarer_Direction are. Or perhaps as a comparison sanity check.
    # self.df = self.df.with_columns(
    #     # easier to use discrete replaces instead of having to slice contract (nt, pass would be a complication)
    #     # first NT->N and suit symbols to SHDCN
    #     # If BidLvl is None, make Contract None
    #     pl.when(pl.col('BidLvl').is_null())
    #     .then(None)
    #     .otherwise(pl.col('BidLvl').cast(pl.String)+pl.col('BidSuit')+pl.col('Dbl')+pl.col('Declarer_Direction'))
    #     .alias('Contract'),
    # )
    return df['Contract'].str.to_uppercase().str.replace('♠','S').str.replace('♥','H').str.replace('♦','D').str.replace('♣','C').str.replace('NT','N')


# None is used instead of pl.Null because pl.Null becomes 'Null' string in pl.String columns. Not sure what's going on but the solution is to use None.
def extract_declarer_from_contract(df: pl.DataFrame) -> pl.DataFrame:
    """Extract declarer direction from contract string using vectorized operations.

    Purpose:
    - Parse the declarer position (N/E/S/W) from contract notation
    - Handle PASS contracts and null values appropriately
    - Use efficient Polars vectorized string operations

    Parameters:
    - df: DataFrame containing contract information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Declarer_Direction' column

    Input columns:
    - 'Contract': Contract string (e.g., '1NTN', '3SS', 'PASS')

    Output columns:
    - 'Declarer_Direction': Single character direction (N/E/S/W) or None for PASS
    """
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .otherwise(pl.col('Contract').str.slice(-1))
        .alias('Declarer_Direction')
    )


def extract_declarer_pair(df: pl.DataFrame) -> pl.DataFrame:
    """Extract declarer's partnership (NS/EW) from contract using vectorized operations.

    Purpose:
    - Determine which partnership (North-South or East-West) the declarer belongs to
    - Handle PASS contracts and null values appropriately
    - Use efficient Polars vectorized operations

    Parameters:
    - df: DataFrame containing contract information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Declarer_Pair_Direction' column

    Input columns:
    - 'Contract': Contract string containing declarer position

    Output columns:
    - 'Declarer_Pair_Direction': Partnership direction ('NS' or 'EW') or None for PASS
    """
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .when(pl.col('Contract').str.slice(-1).is_in(['N', 'S']))
        .then(pl.lit('NS'))
        .otherwise(pl.lit('EW'))
        .alias('Declarer_Pair_Direction')
    )


def extract_vulnerable_declarer(df: pl.DataFrame) -> pl.DataFrame:
    """Determine declarer's vulnerability status using vectorized operations.

    Purpose:
    - Calculate whether the declarer's partnership is vulnerable for scoring
    - Combine declarer direction with board vulnerability conditions
    - Handle PASS contracts and null values appropriately

    Parameters:
    - df: DataFrame containing contract and vulnerability information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Vul_Declarer' column

    Input columns:
    - 'Contract': Contract string containing declarer position
    - 'Declarer_Pair_Direction': Partnership direction (NS/EW)
    - 'Vul_NS': North-South vulnerability flag
    - 'Vul_EW': East-West vulnerability flag

    Output columns:
    - 'Vul_Declarer': Boolean indicating if declarer's partnership is vulnerable
    """
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .when(pl.col('Declarer_Pair_Direction').eq('NS'))
        .then(pl.col('Vul_NS'))
        .when(pl.col('Declarer_Pair_Direction').eq('EW'))
        .then(pl.col('Vul_EW'))
        .alias('Vul_Declarer')
    )

def extract_contract_level(df: pl.DataFrame) -> pl.DataFrame:
    """Extract bid level (1-7) from contract string using vectorized operations.

    Purpose:
    - Parse the level portion of bridge contracts
    - Convert string level to numeric format for calculations
    - Handle PASS contracts and invalid formats

    Parameters:
    - df: DataFrame containing contract information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'BidLvl' column

    Input columns:
    - 'Contract': Contract string (e.g., '1NT', '3S', '7NTX')

    Output columns:
    - 'BidLvl': Numeric level (1-7) as UInt8, None for PASS or invalid contracts
    """
    return df.with_columns(
        pl.col('Contract').str.slice(0, 1).cast(pl.UInt8,strict=False).alias('BidLvl')
    )

def extract_contract_strain(df: pl.DataFrame) -> pl.DataFrame:
    """Extract bid strain (S/H/D/C/N) from contract string using vectorized operations.

    Purpose:
    - Parse the suit/strain portion of bridge contracts
    - Extract second character representing trump suit or no-trump
    - Handle PASS contracts and null values

    Parameters:
    - df: DataFrame containing contract information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'BidSuit' column

    Input columns:
    - 'Contract': Contract string (e.g., '1NT', '3S', '7C')

    Output columns:
    - 'BidSuit': Single character strain (S/H/D/C/N) or None for PASS
    """
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .otherwise(pl.col('Contract').str.slice(1, 1))
        .alias('BidSuit')
    )


def extract_contract_doubling(df: pl.DataFrame) -> pl.DataFrame:
    """Extract doubling status from contract string using vectorized operations.

    Purpose:
    - Parse doubling information (undoubled, doubled, redoubled) from contracts
    - Determine doubling based on contract string length
    - Handle PASS contracts and various contract formats

    Parameters:
    - df: DataFrame containing contract information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Dbl' column

    Input columns:
    - 'Contract': Contract string (e.g., '1NT', '3SX', '7NTXX')

    Output columns:
    - 'Dbl': Doubling status ('' for undoubled, 'X' for doubled, 'XX' for redoubled)
    """
    x = df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(pl.lit(None))
        .when(pl.col('Contract').str.len_chars().eq(3))
        .then(pl.lit(''))
        .when(pl.col('Contract').str.len_chars().eq(4))
        .then(pl.lit('X'))
        .when(pl.col('Contract').str.len_chars().eq(5))
        .then(pl.lit('XX'))
        .alias('Dbl')
    )
    logger.info(x['Dbl'].value_counts().sort('Dbl'))
    return x

def get_declarer_name(df: pl.DataFrame) -> pl.DataFrame:
    """Extract declarer's name based on declarer direction using pure Polars operations.

    Purpose:
    - Map declarer direction (N/E/S/W) to the corresponding player name
    - Create declarer identification for scoring and analysis
    - Handle all four possible declarer positions efficiently

    Parameters:
    - df: DataFrame containing player names and declarer direction

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Declarer_Name' column

    Input columns:
    - 'Declarer_Direction': Single character direction (N/E/S/W)
    - 'Player_Name_N': North player name
    - 'Player_Name_E': East player name
    - 'Player_Name_S': South player name
    - 'Player_Name_W': West player name

    Output columns:
    - 'Declarer_Name': Name of the declarer player
    """
    return df.with_columns(
        pl.when(pl.col('Declarer_Direction').eq('N')).then(pl.col('Player_Name_N'))
        .when(pl.col('Declarer_Direction').eq('E')).then(pl.col('Player_Name_E'))
        .when(pl.col('Declarer_Direction').eq('S')).then(pl.col('Player_Name_S'))
        .when(pl.col('Declarer_Direction').eq('W')).then(pl.col('Player_Name_W'))
        .otherwise(None)
        .alias('Declarer_Name')
    )


def get_declarer_id(df: pl.DataFrame) -> pl.DataFrame:
    """Extract declarer's ID based on declarer direction using pure Polars operations.

    Purpose:
    - Map declarer direction (N/E/S/W) to the corresponding player ID
    - Create declarer identification for player tracking and statistics
    - Handle all four possible declarer positions efficiently

    Parameters:
    - df: DataFrame containing player IDs and declarer direction

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Declarer_ID' column

    Input columns:
    - 'Declarer_Direction': Single character direction (N/E/S/W)
    - 'Player_ID_N': North player ID
    - 'Player_ID_E': East player ID
    - 'Player_ID_S': South player ID
    - 'Player_ID_W': West player ID

    Output columns:
    - 'Declarer_ID': ID of the declarer player
    """
    return df.with_columns(
        pl.when(pl.col('Declarer_Direction').eq('N')).then(pl.col('Player_ID_N'))
        .when(pl.col('Declarer_Direction').eq('E')).then(pl.col('Player_ID_E'))
        .when(pl.col('Declarer_Direction').eq('S')).then(pl.col('Player_ID_S'))
        .when(pl.col('Declarer_Direction').eq('W')).then(pl.col('Player_ID_W'))
        .otherwise(None)
        .alias('Declarer_ID')
    )


# convert to ml df needs to perform this.
def compute_contract_result(df: pl.DataFrame) -> list[Optional[int]]:
    """Compute contract results by matching actual scores against expected score lists.

    Purpose:
    - Calculate bridge contract results by finding score position in lookup table
    - Handle complex scoring scenarios with doubling and vulnerability
    - Legacy implementation using mixed Python/Polars operations

    Parameters:
    - df: DataFrame containing contract and scoring information
    
    Returns:
    - list[Optional[int]]: Contract results for each row (tricks over/under contract)

    Input columns:
    - 'Contract': Contract string (must not be 'PASS')
    - 'Score_NS': Actual North-South score achieved
    - 'scores_l': List of possible scores for the contract
    - 'BidLvl': Bid level for calculation offset

    Output columns:
    - Returns list for 'Result' column (not added to DataFrame directly)

    Note:
    - Legacy implementation - should be refactored to pure Polars
    - Takes ~2 minutes for large datasets, 25x faster with caching
    """
    # note: legacy implementation mixing Python and Polars.
    # todo: refactor to compute Result via pure-Polars using a precomputed score lookup (no .rows()).
    # takes 2m. 25x faster using cache. Cache is about 600 scores lists.
    # convert Score_NS into Result (-13 to 13) and Tricks (0 to 13)
    result_col = 'Result' if 'Result' not in df.columns else 'Result2'
    scores_l_cache = {}
    df = df.with_columns(pl.Series('scores_l',mlBridgeLib.ContractToScores(df,cache=scores_l_cache),dtype=pl.List(pl.Int16)))
    logger.info(len(scores_l_cache))
    logger.info(df[['Contract','BidLvl','BidSuit','Dbl','Vul','Declarer_Direction','Score_NS','scores_l']])
    # todo: can map_elements be eliminated?
    df = df.with_columns(
        pl.struct(['scores_l', 'Score_NS', 'BidLvl']).map_elements(
            lambda row: None if row['BidLvl'] is None or row['Score_NS'] not in row['scores_l'] else row['scores_l'].index(row['Score_NS']) - (row['BidLvl'] + 6),
            return_dtype=pl.Int8
        ).alias(result_col)
    )
    if result_col == 'Result2':
        logger.info(df[['Contract','BidLvl','BidSuit','Dbl','Vul','Declarer_Direction','Score_NS','scores_l','Result','Result2']])
        assert df['Result'].eq(df['Result2']).all()
    else:
        logger.info(df[['Contract','BidLvl','BidSuit','Dbl','Vul','Declarer_Direction','Score_NS','scores_l','Result']])
    return df


def compute_result_from_tricks(df: pl.DataFrame) -> pl.DataFrame:
    """Compute contract result from actual tricks taken using vectorized operations.

    Purpose:
    - Calculate bridge contract results (made/failed by how many tricks)
    - Convert raw trick count to standard bridge result notation
    - Handle PASS contracts and missing data appropriately

    Parameters:
    - df: DataFrame containing contract and trick information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'Result' column

    Input columns:
    - 'Tricks': Number of tricks actually taken by declarer (0-13)
    - 'Contract': Contract string for level extraction
    - 'BidLvl': Contract level (alternative to parsing Contract)

    Output columns:
    - 'Result': Contract result relative to bid (+3 for making 3 over, -2 for down 2)
    """
    return df.with_columns(
        pl.when(pl.col('Tricks').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .otherwise(pl.col('Tricks') - 6 - pl.col('Contract').str.replace('PASS', '0').str.slice(0, 1).cast(pl.UInt8)) # note that replace('PASS', '0') must be used because otherwise works on ALL rows.
        .alias('Result')
    )


def compute_expected_tricks(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate expected tricks needed for contract based on bid level using vectorized operations.

    Purpose:
    - Compute the minimum tricks required to make a bridge contract
    - Add book (6 tricks) to bid level for total expected tricks
    - Handle PASS contracts and missing data appropriately

    Parameters:
    - df: DataFrame containing contract information
    
    Returns:
    - pl.DataFrame: Input DataFrame with added 'Expected_Tricks' column

    Input columns:
    - 'Contract': Contract string for level extraction
    - 'Result': Contract result (used for null checking)

    Output columns:
    - 'Expected_Tricks': Total tricks needed to make contract (bid level + 6)
    """
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS') | pl.col('Result').is_null())
        .then(None)
        .otherwise(pl.col('Contract').str.replace('PASS', '0').str.slice(0, 1).cast(pl.UInt8) + 6 + pl.col('Result')) # note that replace('PASS', '0') must be used because otherwise works on ALL rows.
        .alias('Tricks')
    )


# def convert_contract_to_DD_Tricks(df: pl.DataFrame) -> list[Optional[int]]:
#     return [None if c is None or c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Declarer_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


# def convert_contract_to_DD_Tricks_Dummy(df: pl.DataFrame) -> list[Optional[int]]:
#     return [None if c is None or c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Dummy_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


def add_dd_tricks_column(df: pl.DataFrame) -> pl.DataFrame:
    """Create DD_Tricks column by extracting double dummy tricks for the declarer using optimized vectorized operations.

    Purpose:
    - Extract predicted tricks from double dummy analysis for the declarer
    - Map declarer direction and bid suit to corresponding DD columns
    - Provide optimized Polars implementation for performance

    Parameters:
    - df: DataFrame containing double dummy data and contract information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'DD_Tricks' column

    Input columns:
    - 'Declarer_Direction': Declarer seat (N/E/S/W)
    - 'BidSuit': Contract strain (S/H/D/C/N)
    - 'DD_{direction}_{strain}': Double dummy trick columns for all direction/strain combinations

    Output columns:
    - 'DD_Tricks': Predicted tricks the declarer can take in their contract strain
    """
    return df.with_columns(
        pl.struct(['Declarer_Direction', 'BidSuit', pl.col('^DD_[NESW]_[CDHSN]$')])
        .map_elements(
            lambda r: None if r['Declarer_Direction'] is None else r[f"DD_{r['Declarer_Direction']}_{r['BidSuit']}"],
            return_dtype=pl.UInt8
        )
        .alias('DD_Tricks')
    )


def add_dd_tricks_dummy_column(df: pl.DataFrame) -> pl.DataFrame:
    """Create DD_Tricks_Dummy column by extracting double dummy tricks for the dummy using optimized vectorized operations.

    Purpose:
    - Extract predicted tricks from double dummy analysis for the dummy player
    - Map dummy direction and bid suit to corresponding DD columns
    - Provide optimized Polars implementation for performance

    Parameters:
    - df: DataFrame containing double dummy data and contract information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'DD_Tricks_Dummy' column

    Input columns:
    - 'Dummy_Direction': Dummy seat (N/E/S/W)
    - 'BidSuit': Contract strain (S/H/D/C/N)
    - 'DD_{direction}_{strain}': Double dummy trick columns for all direction/strain combinations

    Output columns:
    - 'DD_Tricks_Dummy': Predicted tricks the dummy can take in the contract strain
    """
    return df.with_columns(
        pl.struct(['Dummy_Direction', 'BidSuit', pl.col('^DD_[NESW]_[CDHSN]$')])
        .map_elements(
            lambda r: None if r['Dummy_Direction'] is None else r[f"DD_{r['Dummy_Direction']}_{r['BidSuit']}"],
            return_dtype=pl.UInt8
        )
        .alias('DD_Tricks_Dummy')
    )


# todo: implement this
def augment_acbl_hand_records(df: pl.DataFrame) -> pl.DataFrame:
    """Comprehensive augmentation of ACBL hand records with all bridge analysis features.

    Purpose:
    - Apply complete suite of bridge data augmentations for ACBL hand records
    - Add contract analysis, scoring, player information, and temporal features
    - Create standardized Date column from game_date if available

    Parameters:
    - df: DataFrame containing raw ACBL hand record data

    Returns:
    - pl.DataFrame: Fully augmented DataFrame with all bridge analysis columns

    Input columns:
    - Core hand record data including contracts, players, scores
    - 'game_date': Optional datetime string for date parsing

    Output columns:
    - All contract analysis columns (declarer, level, strain, etc.)
    - All scoring columns (tricks, results, matchpoints, etc.)
    - All player analysis columns (names, IDs, ratings, etc.)
    - 'Date': Parsed date column if game_date provided
    """

    augmenter = FinalContractAugmenter(df)
    df = augmenter.perform_final_contract_augmentations()

    # takes 5s
    if 'game_date' in df.columns:
        t = time.time()
        df = df.with_columns(pl.Series('Date',df['game_date'].str.strptime(pl.Date,'%Y-%m-%d %H:%M:%S')))
        logger.info(f"Time to create ACBL Date: {time.time()-t} seconds")
    # takes 5s
    if 'hand_record_id' in df.columns:
        t = time.time()
        df = df.with_columns(
            pl.col('hand_record_id').cast(pl.String),
        )
        logger.info(f"Time to create ACBL hand_record_id: {time.time()-t} seconds")
    return df

# Global column creation functions

def add_distribution_points(df: pl.DataFrame) -> pl.DataFrame:
    """Create DP_[NESW]_[SHDC] columns from SL_[NESW]_[SHDC] columns.

    Purpose:
    - Calculate distribution points for bridge hand evaluation
    - Convert suit length patterns to standard distribution scoring
    - Support hand strength assessment beyond high card points
    - Generate distributional values: void=3, singleton=2, doubleton=1, 3+=0

    Parameters:
    - df: DataFrame containing suit length information

    Input columns:
    - `SL_{NESW}_{SHDC}`: Suit length counts for each direction and suit

    Output columns:
    - `DP_{NESW}_{SHDC}`: Distribution points by direction and suit (pl.UInt8)

    Returns:
    - DataFrame with added distribution point columns
    """
    dp_columns = [
        pl.when(pl.col(f"SL_{direction}_{suit}") == 0).then(3)
        .when(pl.col(f"SL_{direction}_{suit}") == 1).then(2)
        .when(pl.col(f"SL_{direction}_{suit}") == 2).then(1)
        .otherwise(0)
        .alias(f"DP_{direction}_{suit}")
        for direction in "NESW"
        for suit in "SHDC"
    ]
    return df.with_columns(dp_columns)
def add_total_points(df: pl.DataFrame) -> pl.DataFrame:
    """Create Total_Points columns from HCP and DP columns.

    Purpose:
    - Calculate combined hand strength using both high card points and distribution points
    - Generate comprehensive hand evaluation metrics for bridge analysis
    - Sum HCP and distribution points at suit, direction, and partnership levels
    - Support modern hand evaluation methods combining honor strength and distribution

    Parameters:
    - df: DataFrame containing HCP and distribution point information

    Input columns:
    - `HCP_{NESW}_{SHDC}`: High card points by direction and suit
    - `DP_{NESW}_{SHDC}`: Distribution points by direction and suit

    Output columns:
    - `Total_Points_{NESW}_{SHDC}`: Combined points by direction and suit (pl.UInt8)
    - `Total_Points_{NESW}`: Total points by direction (pl.UInt8)
    - `Total_Points_{NS|EW}`: Partnership total points (pl.UInt8)

    Returns:
    - DataFrame with added total point columns
    """
    # Individual suit total points
    df = df.with_columns([
        (pl.col(f'HCP_{d}_{s}')+pl.col(f'DP_{d}_{s}')).alias(f'Total_Points_{d}_{s}')
        for d in 'NESW'
        for s in 'SHDC'
    ])
    
    # Direction total points
    df = df.with_columns([
        (pl.sum_horizontal([f'Total_Points_{d}_{s}' for s in 'SHDC'])).alias(f'Total_Points_{d}')
        for d in 'NESW'
    ])
    
    # Pair total points
    df = df.with_columns([
        (pl.col('Total_Points_N')+pl.col('Total_Points_S')).alias('Total_Points_NS'),
        (pl.col('Total_Points_E')+pl.col('Total_Points_W')).alias('Total_Points_EW'),
    ])
    
    return df

def add_contract_type(df: pl.DataFrame) -> pl.DataFrame:
    """Create ContractType column from Contract, BidLvl, and BidSuit columns.

    Purpose:
    - Classify bridge contracts into standard categories (Pass, Partial, Game, Slam)
    - Support contract-type-based analysis and statistics
    - Enable filtering and grouping by contract difficulty/type
    - Generate readable contract categorization for reporting

    Parameters:
    - df: DataFrame containing contract bid information

    Input columns:
    - `Contract`: Contract string (for PASS detection)
    - `BidLvl`: Bid level (1-7) 
    - `BidSuit`: Bid suit (C/D/H/S/N)

    Output columns:
    - `ContractType`: Contract category string (Pass/Partial/Game/SSlam/GSlam) (pl.String)

    Returns:
    - DataFrame with added ContractType column
    """
    return df.with_columns(
        pl.when(pl.col('Contract').eq('PASS')).then(pl.lit("Pass"))
        .when(pl.col('BidLvl').eq(5) & pl.col('BidSuit').is_in(['C', 'D'])).then(pl.lit("Game"))
        .when(pl.col('BidLvl').is_in([4,5]) & pl.col('BidSuit').is_in(['H', 'S'])).then(pl.lit("Game"))
        .when(pl.col('BidLvl').is_in([3,4,5]) & pl.col('BidSuit').eq('N')).then(pl.lit("Game"))
        .when(pl.col('BidLvl').eq(6)).then(pl.lit("SSlam"))
        .when(pl.col('BidLvl').eq(7)).then(pl.lit("GSlam"))
        .otherwise(pl.lit("Partial"))
        .alias('ContractType')
    )

def add_player_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Create Player_ID_N, Player_ID_S, Player_ID_E, Player_ID_W columns from Pair_IDs_NS and Pair_IDs_EW.

    Purpose:
    - Generate individual player ids for display and analysis
    - Create standardized player identification for each player
    - Support player-based reporting and statistics
    - Enable player tracking across multiple sessions and events

    Parameters:
    - df: DataFrame containing individual player ids

    Input columns:
    - `Player_ID_N`: North player id
    - `Player_ID_S`: South player id  
    - `Player_ID_E`: East player id
    - `Player_ID_W`: West player id

    Output columns:
    - `Player_ID_N`: North player id (pl.String)
    - `Player_ID_S`: South player id (pl.String)
    - `Player_ID_E`: East player id (pl.String)
    - `Player_ID_W`: West player id (pl.String)

    Returns:
    - DataFrame with added individual player id columns
    """
    return df.with_columns([
            pl.col("Pair_IDs_NS").list.get(0).alias("Player_ID_N"),
            pl.col("Pair_IDs_NS").list.get(1).alias("Player_ID_S"),
            pl.col("Pair_IDs_EW").list.get(0).alias("Player_ID_E"),
            pl.col("Pair_IDs_EW").list.get(1).alias("Player_ID_W"),
        ])

def add_player_names(df: pl.DataFrame) -> pl.DataFrame:
    """Create Player_Name_N, Player_Name_S, Player_Name_E, Player_Name_W columns from Pair_Names_NS and Pair_Names_EW.

    Purpose:
    - Generate individual player names for display and analysis
    - Create standardized player identification for each player
    - Support player-based reporting and statistics
    - Enable player tracking across multiple sessions and events

    Parameters:
    - df: DataFrame containing individual player names

    Input columns:
    - `Player_Name_NS`: North South player name
    - `Player_Name_EW`: East West player name

    Output columns:
    - `Player_Name_N`: North player name (pl.String)
    - `Player_Name_S`: South player name (pl.String)
    - `Player_Name_E`: East player name (pl.String)
    - `Player_Name_W`: West player name (pl.String)

    Returns:
    - DataFrame with added individual player name columns
    """
    return df.with_columns([
        pl.col("Player_Name_NS").list.get(0).alias("Player_Name_N"),
        pl.col("Player_Name_NS").list.get(1).alias("Player_Name_S"),
        pl.col("Player_Name_EW").list.get(0).alias("Player_Name_E"),
        pl.col("Player_Name_EW").list.get(1).alias("Player_Name_W"),
    ])

def add_pair_names(df: pl.DataFrame) -> pl.DataFrame:
    """Create Pair_Names_(NS|EW) from Player_Name_(N|S|E|W).

    Purpose:
    - Generate partnership player name combinations for display and analysis
    - Create standardized player identification for North-South and East-West pairs
    - Support player-based reporting and statistics
    - Enable player tracking across multiple sessions and events

    Parameters:
    - df: DataFrame containing individual player names

    Input columns:
    - `Player_Name_N`: North player name
    - `Player_Name_S`: South player name  
    - `Player_Name_E`: East player name
    - `Player_Name_W`: West player name

    Output columns:
    - `Pair_Names_NS`: North-South partnership player names (pl.String)
    - `Pair_Names_EW`: East-West partnership player names (pl.String)

    Returns:
    - DataFrame with added partnership player name columns
    """
    return df.with_columns([
        pl.concat_list([pl.col("Player_Name_N"), pl.col("Player_Name_S")]).alias("Pair_Names_NS"),
        pl.concat_list([pl.col("Player_Name_E"), pl.col("Player_Name_W")]).alias("Pair_Names_EW"),
    ])

def add_pair_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Create Pair_IDs_(NS|EW) from Player_ID_(N|S|E|W).

    Purpose:
    - Generate partnership player ids combinations for display and analysis
    - Create standardized player identification for North-South and East-West pairs
    - Support player-based reporting and statistics
    - Enable player tracking across multiple sessions and events

    Parameters:
    - df: DataFrame containing individual player ids

    Input columns:
    - `Player_ID_N`: North player id
    - `Player_ID_S`: South player id  
    - `Player_ID_E`: East player id
    - `Player_ID_W`: West player id

    Output columns:
    - `Pair_IDs_NS`: North-South partnership player ids (pl.String)
    - `Pair_IDs_EW`: East-West partnership player ids (pl.String)

    Returns:
    - DataFrame with added partnership player id columns
    """
    return df.with_columns([
        pl.concat_list([pl.col("Player_ID_N"), pl.col("Player_ID_S")]).alias("Pair_IDs_NS"),
        pl.concat_list([pl.col("Player_ID_E"), pl.col("Player_ID_W")]).alias("Pair_IDs_EW"),
    ])

def add_declarer_info(df: pl.DataFrame) -> pl.DataFrame:
    """Create Declarer_Name and Declarer_ID columns from Declarer_Direction, Player_ID_(N|S|E|W), Player_Name_(N|S|E|W).

    Purpose:
    - Extract declarer identification information from contract and player data
    - Create declarer-specific columns for performance analysis and tracking
    - Support declarer-focused statistics and reporting
    - Enable player performance analysis by declarer position

    Parameters:
    - df: DataFrame containing contract and player information

    Input columns:
    - `Declarer_Direction`: Declarer position (N/E/S/W)
    - Player ID and name columns (processed by get_declarer_name and get_declarer_id)

    Output columns:
    - `Declarer_Name`: Name of the declarer player (pl.String)
    - `Declarer_ID`: ID of the declarer player (appropriate ID type)

    Returns:
    - DataFrame with added declarer identification columns
    """
    df = get_declarer_name(df)
    df = get_declarer_id(df)
    return df


# =============================================================================
# SECTION 6: SCORING SYSTEM FUNCTIONS
# =============================================================================
# Used by: MatchPointAugmenter, IMPAugmenter
# Purpose: Matchpoint and IMP calculations, percentage scores, ranking systems
# Temporal order: Sixth - these compute competitive scoring metrics
# =============================================================================

def add_board_matchpoint_top(df: pl.DataFrame) -> pl.DataFrame:
    """Create matchpoint top (maximum possible) for each board-session combination.

    Purpose:
    - Calculate the maximum matchpoint score available for each board in a session
    - Determine board-specific matchpoint scaling based on number of tables
    - Support matchpoint percentage calculations

    Parameters:
    - df: DataFrame containing board and session information

    Returns:
    - pl.DataFrame: Input DataFrame with added 'MP_Top' column

    Input columns:
    - 'MP_NS': North-South matchpoint score
    - 'MP_EW': East-West matchpoint score

    Output columns:
    - 'MP_Top': Maximum matchpoint score for each board in the session
    """
    df = df.with_columns(
        pl.col("MP_NS").add(pl.col('MP_EW')).cast(pl.Float64).round(0).cast(pl.UInt32).alias("MP_Top"),
        # pl.col('Score').count().over(['session_id','PBN','Board']).sub(1).alias('MP_Top') # acceptable alternative?
    )
    # filtering out bad data. Only in club data?
    # print(df.filter(pl.col('MP_Top').is_null())['MP_Top'])
    df =  df.filter(pl.col('MP_Top').is_not_null() & pl.col('MP_Top').gt(0))
    return df

def add_matchpoint_scores_from_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Create matchpoint scores for both partnerships from their raw scores.

    Purpose:
    - Convert raw bridge scores to matchpoint scores for tournament ranking
    - Calculate comparative performance against all other pairs on same boards
    - Generate both North-South and East-West matchpoint perspectives

    Parameters:
    - df: DataFrame containing raw scores and grouping keys

    Returns:
    - pl.DataFrame: Input DataFrame with added matchpoint score columns

    Grouping keys priority (used to define the comparison cohort per board):
    - Always: 'session_id', 'Board'
    - Prefer: 'section_id' if present, else 'section_name' if present
    - Optionally include 'PBN' if present (for extra precision) but not required

    Output columns:
    - 'MP_NS': Matchpoint score for North-South pair
    - 'MP_EW': Matchpoint score for East-West pair (complement of MP_NS)
    """
    # Determine grouping keys dynamically to avoid hard dependency on PBN
    group_keys: list[str] = ['session_id', 'Board']
    if 'section_id' in df.columns:
        group_keys = ['session_id', 'section_id', 'Board']
    elif 'section_name' in df.columns:
        group_keys = ['session_id', 'section_name', 'Board']
    # If PBN exists, include it for tighter grouping; otherwise skip
    if 'PBN' in df.columns and 'PBN' not in group_keys: # todo: is this now obsolete? isn't session_id enough? what's the compute cost of including PBN?
        # Insert before Board to keep stable order
        group_keys = [k for k in group_keys if k != 'Board'] + ['PBN', 'Board']

    return df.with_columns([
        pl.col('Score_NS').rank(method='average', descending=False).sub(1)
            .over(group_keys).alias('MP_NS'),
        pl.col('Score_EW').rank(method='average', descending=False).sub(1)
            .over(group_keys).alias('MP_EW'),
    ])

def add_percentage_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Create Pct_NS and Pct_EW columns from MP_NS, MP_EW, and MP_Top columns.

    Purpose:
    - Convert raw matchpoint scores to percentage values for standardized analysis
    - Calculate partnership percentages for performance comparison
    - Enable percentage-based ranking and statistical analysis
    - Normalize scores across different field sizes and session formats

    Parameters:
    - df: DataFrame containing matchpoint score information

    Input columns:
    - `MP_NS`: North-South raw matchpoint score
    - `MP_EW`: East-West raw matchpoint score  
    - `MP_Top`: Maximum possible matchpoints for the session

    Output columns:
    - `Pct_NS`: North-South percentage score (pl.Float32)
    - `Pct_EW`: East-West percentage score (pl.Float32)

    Returns:
    - DataFrame with added percentage score columns
    """
    df = df.with_columns([
        (pl.col('MP_NS') / pl.col('MP_Top')).cast(pl.Float32).alias('Pct_NS'),
        (pl.col('MP_EW') / pl.col('MP_Top')).cast(pl.Float32).alias('Pct_EW')
    ])
    # df =  df.filter(~pl.col('Pct_NS').is_infinite() & ~pl.col('Pct_EW').is_infinite())
    return df

def add_declarer_percentage(df: pl.DataFrame) -> pl.DataFrame:
    """Create Declarer_Pct column from Declarer_Pair_Direction, Pct_NS, and Pct_EW columns.

    Purpose:
    - Extract the declaring partnership's percentage score for declarer-focused analysis
    - Map declarer pair direction to the appropriate percentage score
    - Enable declarer performance tracking and evaluation
    - Support declarer-specific statistical analysis and reporting

    Parameters:
    - df: DataFrame containing declarer direction and partnership percentages

    Input columns:
    - `Declarer_Pair_Direction`: Partnership of the declarer (NS/EW)
    - `Pct_NS`: North-South percentage score
    - `Pct_EW`: East-West percentage score

    Output columns:
    - `Declarer_Pct`: Percentage score of the declaring partnership (pl.Float32)

    Returns:
    - DataFrame with added declarer percentage column
    """
    return df.with_columns(
        pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
        .then('Pct_NS')
        .otherwise('Pct_EW')
        .alias('Declarer_Pct')
    )

def add_max_suit_lengths(df: pl.DataFrame) -> pl.DataFrame:
    """Create SL_Max_[NS|EW] columns from SL_[NS|EW]_[SHDC] columns using vectorized operations.

    Purpose:
    - Identify the longest suit for each partnership (NS/EW)
    - Calculate maximum suit length for distribution analysis
    - Determine which suit contains the partnership's longest holding
    - Support bid evaluation and hand strength assessment

    Parameters:
    - df: DataFrame containing partnership suit length information

    Input columns:
    - `SL_{NS|EW}_{SHDC}`: Partnership suit lengths for each suit

    Output columns:
    - `SL_Max_{NS|EW}`: Maximum suit length for each partnership (pl.UInt8)
    - `SL_Max_{NS|EW}_Col`: Column name of the suit with maximum length (pl.String)

    Returns:
    - DataFrame with added maximum suit length columns
    """
    # Create max suit length columns for each pair direction
    for pair_direction in ['NS', 'EW']:
        # Get the suit length columns for this pair
        suit_cols = [f"SL_{pair_direction}_{suit}" for suit in 'SHDC']
        
        # Find the maximum suit length and its corresponding suit
        max_expr = pl.max_horizontal(suit_cols).alias(f'SL_Max_{pair_direction}')
        
        # Find which suit has the maximum length
        max_col_expr = (
            pl.when(pl.col(suit_cols[0]) == pl.max_horizontal(suit_cols)).then(pl.lit(suit_cols[0]))
            .when(pl.col(suit_cols[1]) == pl.max_horizontal(suit_cols)).then(pl.lit(suit_cols[1]))
            .when(pl.col(suit_cols[2]) == pl.max_horizontal(suit_cols)).then(pl.lit(suit_cols[2]))
            .otherwise(pl.lit(suit_cols[3]))
            .alias(f'SL_Max_{pair_direction}_Col')
        )
        
        df = df.with_columns([max_expr, max_col_expr])
    
    return df

def add_suit_quality_indicators(df: pl.DataFrame, suit_quality_criteria: dict, stopper_criteria: dict) -> pl.DataFrame:
    """Create quality indicator columns from SL and HCP columns.

    Purpose:
    - Generate suit quality assessments based on length and high card points
    - Identify suits meeting specific criteria for bidding and play decisions
    - Create boolean indicators for suit evaluation in bridge analysis
    - Support strategic planning based on suit quality metrics

    Parameters:
    - df: DataFrame containing suit length and HCP information
    - suit_quality_criteria: Dictionary defining criteria for suit quality assessment
    - stopper_criteria: Dictionary defining criteria for stopper identification

    Input columns:
    - `SL_{NESW}_{SHDC}`: Suit length by direction and suit
    - `HCP_{NESW}_{SHDC}`: High card points by direction and suit

    Output columns:
    - Quality indicator columns based on criteria functions (pl.Boolean)

    Returns:
    - DataFrame with added suit quality indicator columns
    """
    series_expressions = [
        pl.Series(
            f"{series_type}_{direction}_{suit}",
            criteria(
                df[f"SL_{direction}_{suit}"],
                df[f"HCP_{direction}_{suit}"]
            ),
            pl.Boolean
        )
        for direction in "NESW"
        for suit in "SHDC"
        for series_type, criteria in {**suit_quality_criteria, **stopper_criteria}.items()
    ]
    
    df = df.with_columns(series_expressions)
    df = df.with_columns([
        pl.lit(False).alias(f"Forcing_One_Round"),
        pl.lit(False).alias(f"Opponents_Cannot_Play_Undoubled_Below_2N"),
        pl.lit(False).alias(f"Forcing_To_2N"),
        pl.lit(False).alias(f"Forcing_To_3N"),
    ])
    return df

def add_balanced_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Create Balanced_[NESW] columns from SL_[NESW]_ML_SJ, SL_[NESW]_C, and SL_[NESW]_D columns.

    Purpose:
    - Identify balanced hand distributions for bridge analysis
    - Create boolean indicators for balanced vs unbalanced hands
    - Support hand type classification for bidding and play decisions
    - Enable statistical analysis of hand distribution patterns

    Parameters:
    - df: DataFrame containing suit length and distribution information

    Input columns:
    - `SL_{NESW}_ML_SJ`: Major-length suit distribution pattern strings
    - `SL_{NESW}_C`: Club suit length
    - `SL_{NESW}_D`: Diamond suit length

    Output columns:
    - `Balanced_{NESW}`: Boolean indicators for balanced hand distributions (pl.Boolean)

    Returns:
    - DataFrame with added balanced hand indicator columns
    """
    return df.with_columns([
        pl.Series(
            f"Balanced_{direction}",
            df[f"SL_{direction}_ML_SJ"].is_in(['4-3-3-3','4-4-3-2']) |
            (df[f"SL_{direction}_ML_SJ"].is_in(['5-3-3-2','5-4-2-2']) & 
             (df[f"SL_{direction}_C"].eq(5) | df[f"SL_{direction}_D"].eq(5))),
            pl.Boolean
        )
        for direction in 'NESW'
    ])

def derive_result_from_tricks(df: pl.DataFrame) -> pl.DataFrame:
    """Create Result column from Tricks column using optimized vectorized operations.
    
    Purpose:
    - Compute contract result from actual tricks taken
    - Convert raw trick count to standard bridge result notation
    - Handle PASS contracts and missing data appropriately
    
    Parameters:
    - df: DataFrame containing contract and trick information
    
    Returns:
    - DataFrame with added 'Result' column
    
    Input columns:
    - `Tricks`: Number of tricks actually taken by declarer (0-13)
    - `Contract`: Contract string for level extraction
    - `BidLvl`: Contract level (alternative to parsing Contract)
    
    Output columns:
    - `Result`: Contract result relative to bid (+3 for making 3 over, -2 for down 2)
    """
    return compute_result_from_tricks(df)

def derive_result_from_contract(df: pl.DataFrame) -> pl.DataFrame:
    """Create Result column from Contract column.
    
    Purpose:
    - Calculate bridge contract results (made/failed by how many tricks)
    - Convert raw trick count to standard bridge result notation
    - Handle PASS contracts and missing data appropriately
    
    Parameters:
    - df: DataFrame containing contract and trick information
    
    Returns:
    - DataFrame with added 'Result' column
    
    Input columns:
    - `Tricks`: Number of tricks actually taken by declarer (0-13)
    - `Contract`: Contract string for level extraction
    - `BidLvl`: Contract level (alternative to parsing Contract)
    
    Output columns:
    - `Result`: Contract result relative to bid (+3 for making 3 over, -2 for down 2)
    """
    return compute_contract_result(df)

def derive_tricks_from_contract(df: pl.DataFrame) -> pl.DataFrame:
    """Create Tricks column from Contract column using optimized vectorized operations.
    
    Purpose:
    - Calculate expected tricks needed for contract based on bid level
    - Add book (6 tricks) to bid level for total expected tricks
    - Handle PASS contracts and missing data appropriately
    
    Parameters:
    - df: DataFrame containing contract information
    
    Returns:
    - DataFrame with added 'Tricks' column
    
    Input columns:
    - `Contract`: Contract string for level extraction
    - `Result`: Contract result (used for null checking)
    
    Output columns:
    - `Tricks`: Total tricks needed to make contract (bid level + 6 + result)
    """
    return compute_expected_tricks(df)


def add_dd_scores_basic(df: pl.DataFrame, scores_d: Dict) -> pl.DataFrame:
    """Create basic DD_Score columns for all possible contracts (no contract-dependent columns).
    
    Purpose:
    - Generate double dummy scores for all possible contract combinations
    - Create score lookup tables for contract analysis and evaluation
    - Does not require actual contract information (BidLvl, BidSuit, Declarer_Direction)
    - Provide foundation scoring data for subsequent contract evaluation
    
    Parameters:
    - df: DataFrame containing hand records with DD tricks and vulnerability data
    - scores_d: Dictionary mapping (level, strain, tricks, vulnerability) to scores

    Input columns:
    - `DD_{NESW}_{SHDCN}`: Double-dummy trick counts for each declarer/strain
    - `Vul_NS`, `Vul_EW`: Vulnerability flags for each partnership
    
    Output columns:
    - `DD_Score_{level}{strain}_{NESW}`: Scores for each contract by declarer (pl.Int32)
    - `DD_Score_{level}{strain}_{NS|EW}`: Scores for each contract by partnership (pl.Int32)
    
    Returns:
    - DataFrame with added DD_Score columns for all contract combinations
    
    Input columns:
    - `DD_[NESW]_[CDHSN]`: Double dummy tricks by direction and strain
    - `Vul_(NS|EW)`: Vulnerability flags for partnerships
    
    Output columns:
    - `DD_Score_[1-7][CDHSN]_[NESW]`: Double dummy scores for each contract/direction combo
    - `DD_Score_[CDHSN]_[NESW]_Max`: Maximum double dummy score for each strain/direction combo
    """
    # Create scores for columns: DD_Score_[1-7][CDHSN]_[NESW]
    df = df.with_columns([
        pl.struct([f"DD_{direction}_{strain}", f"Vul_{pair_direction}"])
        .map_elements(
            lambda r, lvl=level, strn=strain, dir=direction, pdir=pair_direction: 
                scores_d.get((lvl, strn, r[f"DD_{dir}_{strn}"], r[f"Vul_{pdir}"]), None),
            return_dtype=pl.Int16
        )
        .alias(f"DD_Score_{level}{strain}_{direction}")
        for level in range(1, 8)
        for strain in mlBridgeLib.CDHSN
        for direction, pair_direction in [('N','NS'), ('E','EW'), ('S','NS'), ('W','EW')]
    ])
    df = df.with_columns([
        pl.max_horizontal(pl.col(f"^DD_Score_[1-7]{strain}_{direction}$")).alias(f"DD_Score_{strain}_{direction}_Max")
        for strain in mlBridgeLib.CDHSN
        for direction in 'NESW'
    ])
    df = df.with_columns([
        pl.max_horizontal(pl.col(f"^DD_Score_{strain}_{pair_direction[0]}_Max$"),pl.col(f"^DD_Score_{strain}_{pair_direction[1]}_Max$")).alias(f"DD_Score_{strain}_{pair_direction}_Max")
        for strain in mlBridgeLib.CDHSN
        for pair_direction in ['NS','EW']
    ])
    
    return df

def add_dd_scores_contract_dependent(df: pl.DataFrame) -> pl.DataFrame:
    """Create contract-dependent DD_Score columns (requires BidLvl, BidSuit, Declarer_Direction).
    
    Purpose:
    - Generate contract-specific double dummy analysis columns
    - Create declarer-centric scoring for actual contracts played
    - Bridge basic DD scores with specific contract outcomes
    - Enable contract-specific performance analysis and comparison
    
    Parameters:
    - df: DataFrame containing contract information and basic DD scores

    Input columns:
    - `BidLvl`: Contract bid level (1-7)
    - `BidSuit`: Contract strain (S/H/D/C/N)
    - `Declarer_Direction`: Declarer position (N/E/S/W)
    - Basic DD_Score columns from add_dd_scores_basic
    
    Output columns:
    - `DD_Score_NS`: Double-dummy score for North-South (pl.Int32)
    - `DD_Score_EW`: Double-dummy score for East-West (pl.Int32)
    - `DD_Score_Declarer`: Double-dummy score from declarer's perspective (pl.Int32)
    
    Returns:
    - DataFrame with added contract-dependent DD scoring columns
    
    Input columns:
    - `BidLvl`: Contract level (1-7)
    - `BidSuit`: Contract strain (S/H/D/C/N)
    - `Declarer_Direction`: Declarer position (N/E/S/W)
    - `Declarer_Pair_Direction`: Declarer partnership (NS/EW)
    - `DD_Score_[1-7][CDHSN]_[NESW]`: Double dummy scores for each contract/direction combo
    - `DD_Score_[CDHSN]_[NESW]_Max`: Double dummy scores for each contract/direction combo

    Output columns:
    - `DD_Score_Refs`: Reference string for DD score lookup
    - `DD_Score_Declarer`: DD score for the actual declarer
    - `DD_Score_NS`: DD score from NS perspective (+/- based on declarer)
    - `DD_Score_EW`: DD score from EW perspective (+/- based on declarer)
    """
    # Create DD_Score_Refs column
    df = df.with_columns(
        (pl.lit('DD_Score_')+pl.col('BidLvl').cast(pl.String)+pl.col('BidSuit')+pl.lit('_')+pl.col('Declarer_Direction')).alias('DD_Score_Refs'),
    )

    # Create list of column names: DD_Score_[1-7][CDHSN]_[NESW]
    dd_score_columns = [f"DD_Score_{level}{strain}_{direction}" 
                        for level in range(1, 8)
                       for strain in 'CDHSN'  
                       for direction in 'NESW']
    
    # Create DD_Score_Declarer by selecting the DD_Score_[1-7][CDHSN]_[NESW] column for the given level, strain, and Declarer_Direction
    df = df.with_columns([
        pl.struct(['BidLvl', 'BidSuit', 'Declarer_Direction'] + dd_score_columns)
        .map_elements(
            lambda r: None if r['Declarer_Direction'] is None else r[f"DD_Score_{r['BidLvl']}{r['BidSuit']}_{r['Declarer_Direction']}"],
            return_dtype=pl.Int16
        )
        .alias('DD_Score_Declarer')
    ])

    # Create list of column names of declarer: DD_Score_[CDHSN]_[NESW]_Max
    dd_score_columns_max = [f"DD_Score_{strain}_{direction}_Max" 
                       for strain in 'CDHSN'  
                       for direction in 'NESW']
    
    # Create DD_Score_Max_Declarer by selecting the DD_Score_[CDHSN]_[NESW]_Max column for the given strain and declarer direction
    df = df.with_columns([
        pl.struct(['BidSuit', 'Declarer_Direction'] + dd_score_columns_max)
        .map_elements(
            lambda r: None if r['Declarer_Direction'] is None else r[f"DD_Score_{r['BidSuit']}_{r['Declarer_Direction']}_Max"],
            return_dtype=pl.Int16
        )
        .alias('DD_Score_Max_Declarer')
    ])    

    # todo:? this is all wrong. probably should be the maximum score that the defenders could have achieved if they had made a further bid.
    # # Create list of column names of defender: DD_Score_[CDHSN]_[NESW]_Max
    # dd_score_columns_defender_max = [f"DD_Score_{strain}_{pair_direction}_Max" 
    #                    for strain in 'CDHSN'  
    #                    for pair_direction in ['NS','EW']]
    
    # # Create DD_Score_Max_Defender by finding the maximum score the defending pair could achieve in ANY strain
    # # This represents the best contract the defenders could have declared if they had won the auction
    # df = df.with_columns([
    #     pl.struct(['Declarer_Direction'] + dd_score_columns_defender_max)
    #     .map_elements(
    #         lambda r: None if r['Declarer_Direction'] is None else max([
    #             r[f"DD_Score_{strain}_{PairDirectionToOpponentPairDirection[PlayerDirectionToPairDirection[r['Declarer_Direction']]]}_Max"] 
    #             for strain in 'CDHSN'
    #         ]),
    #         return_dtype=pl.Int16
    #     )
    #     .alias('DD_Score_Max_Defender')
    # ])

    # Create DD_Score_NS and DD_Score_EW columns
    df = df.with_columns([
        pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
        .then(pl.col('DD_Score_Declarer'))
        .when(pl.col('Declarer_Pair_Direction').eq('EW'))
        .then(pl.col('DD_Score_Declarer').neg())
        .otherwise(0)
        .alias('DD_Score_NS'),
        
        pl.when(pl.col('Declarer_Pair_Direction').eq('EW'))
        .then(pl.col('DD_Score_Declarer'))
        .when(pl.col('Declarer_Pair_Direction').eq('NS'))
        .then(pl.col('DD_Score_Declarer').neg())
        .otherwise(0)
        .alias('DD_Score_EW')
    ])
    
    return df

def add_direction_summaries(df: pl.DataFrame) -> pl.DataFrame:
    """Create direction summary columns (DP_[NESW], DP_[NS|EW], DP_[NS|EW]_[SHDC]) from DP_[NESW]_[SHDC] columns.

    Purpose:
    - Aggregate distribution points at direction and partnership levels
    - Calculate total distributional strength for each seat and pair
    - Create suit-specific distribution summaries for partnerships
    - Support high-level distribution analysis and comparison

    Parameters:
    - df: DataFrame containing individual direction distribution points

    Input columns:
    - `DP_{NESW}_{SHDC}`: Distribution points by direction and suit

    Output columns:
    - `DP_{NESW}`: Total distribution points by direction (pl.UInt8)
    - `DP_{NS|EW}`: Partnership total distribution points (pl.UInt8)
    - `DP_{NS|EW}_{SHDC}`: Partnership distribution points by suit (pl.UInt8)

    Returns:
    - DataFrame with added direction and partnership distribution summary columns
    """
    # Direction total DPs
    df = df.with_columns([
        (pl.col(f'DP_{d}_S')+pl.col(f'DP_{d}_H')+pl.col(f'DP_{d}_D')+pl.col(f'DP_{d}_C')).alias(f'DP_{d}')
        for d in "NESW"
    ])
    
    # Pair total DPs
    df = df.with_columns([
        (pl.col('DP_N')+pl.col('DP_S')).alias('DP_NS'),
        (pl.col('DP_E')+pl.col('DP_W')).alias('DP_EW'),
    ])
    
    # Pair suit DPs
    df = df.with_columns([
        (pl.col(f'DP_N_{s}') + pl.col(f'DP_S_{s}')).alias(f'DP_NS_{s}')
        for s in 'SHDC'
    ] + [
        (pl.col(f'DP_E_{s}') + pl.col(f'DP_W_{s}')).alias(f'DP_EW_{s}')
        for s in 'SHDC'
    ])
    
    return df

# todo: gpt5 suggested: Optional: JIT the loop with numba for 10–100x speedup on large datasets.
def compute_pair_matchpoint_elo_ratings(
    df_sorted: pl.DataFrame,
    *,
    initial_rating: float = 1500.0,
    k_base: float = 24.0,
    provisional_boost_until: int = 100,  # boards per pair-direction
    minimum_sessions: int = 10,  # minimum sessions required for non-None rating
    elo_scale: float = 400.0,
    score_amplifier: float = 1.0,
    replace_existing: bool = True,
) -> pl.DataFrame:
    """Compute Elo-style ratings for pairs on matchpoint events.

    Purpose:
    - Calculate Elo-style skill ratings for bridge pairs based on matchpoint performance
    - Provide leakage-safe rating updates that prevent future information contamination
    - Track rating progression over time with provisional boost for new pairs
    - Generate expected scores and rating deltas for analysis

    Parameters:
    - df_sorted: Sorted DataFrame ('Date','session_id','Round','Board') containing matchpoint game results with pair identifiers
    - initial_rating: starting rating for all entities.
    - k_base: base K-factor.
    - provisional_boost_until: boards threshold for larger early K.
    - minimum_sessions: minimum sessions required for non-None rating (default 10).
    - elo_scale: denominator in logistic expectation (chess default 400.0).
    - score_amplifier: amplifies per-board score around 0.5 (1.0 = no change).
    - replace_existing: if True, drop existing output columns before adding new ones.

    Input columns:
    - `Pair_Number_NS`, `Pair_Number_EW`, `Pct_NS`, `session_id` (and optionally `Section_Pairs`/`Num_Pairs`/`MP_Top`).

    Output columns:
    - `Elo_R_NS_Before`, `Elo_R_EW_Before`: Ratings before the game (pl.Float32, None if Elo_N < minimum_sessions).
    - `Elo_E_NS`, `Elo_E_EW`: Expected scores based on rating difference (pl.Float32).
    - `Elo_R_NS_After`, `Elo_R_EW_After`: Updated ratings after the game (pl.Float32, None if Elo_N < minimum_sessions).
    - `Elo_N_NS`, `Elo_N_EW`: Number of unique sessions played by each pair (pl.Int32).
    - `Elo_Delta_Before`, `Elo_Delta_After`: Rating differences NS minus EW (pl.Float32).
    
    Note: Elo ratings are set to None for pairs with fewer than minimum_sessions sessions played to ensure statistical reliability.

    Returns:
    - Original DataFrame with appended columns: Elo_R/E for NS/EW before/after, counts, deltas.
    """
    # TODO(polars): Consider chunking by session and preparing arrays via Polars
    # group_by to minimize Python overhead; full vectorization of Elo updates is
    # non-trivial due to sequential dependency.
    # Ensure schema
    need_cols = {"Pair_Number_NS", "Pair_Number_EW", "Pct_NS", "session_id"}
    missing = need_cols - set(df_sorted.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_rows = df_sorted.height
    ns_pairs = df_sorted["Pair_Number_NS"].to_numpy()
    ew_pairs = df_sorted["Pair_Number_EW"].to_numpy()
    pct_ns = df_sorted["Pct_NS"].to_numpy()
    session_ids = df_sorted["session_id"].to_numpy()

    # Section-size scaling vector
    scale = np.ones(n_rows, dtype=np.float32)
    if "Section_Pairs" in df_sorted.columns:
        pairs = df_sorted["Section_Pairs"].cast(pl.Float32).to_numpy()
    elif "Num_Pairs" in df_sorted.columns:
        pairs = df_sorted["Num_Pairs"].cast(pl.Float32).to_numpy()
    elif "MP_Top" in df_sorted.columns:
        pairs = (df_sorted["MP_Top"].cast(pl.Float32).to_numpy() / 2.0) + 1.0
    else:
        pairs = None
    if pairs is not None:
        with np.errstate(invalid="ignore"):
            valid = pairs > 1.0
        scale_valid = np.sqrt(np.maximum(pairs[valid] - 1.0, 1.0) / 11.0)
        scale[valid] = scale_valid

    # Map pair ids to contiguous indices per direction
    ns_unique = df_sorted["Pair_Number_NS"].unique().to_list()
    ew_unique = df_sorted["Pair_Number_EW"].unique().to_list()
    ns_index = {pid: i for i, pid in enumerate(ns_unique)}
    ew_index = {pid: i for i, pid in enumerate(ew_unique)}

    ratings_ns = np.full(len(ns_unique), initial_rating, dtype=np.float32)
    ratings_ew = np.full(len(ew_unique), initial_rating, dtype=np.float32)
    boards_played_ns = np.zeros(len(ns_unique), dtype=np.int32)  # Track boards for provisional boost
    boards_played_ew = np.zeros(len(ew_unique), dtype=np.int32)  # Track boards for provisional boost
    
    # Track unique sessions per pair using sets
    sessions_ns = [set() for _ in range(len(ns_unique))]
    sessions_ew = [set() for _ in range(len(ew_unique))]

    # Outputs
    r_ns_before_arr = np.empty(n_rows, dtype=np.float32)
    r_ew_before_arr = np.empty(n_rows, dtype=np.float32)
    e_ns_arr = np.empty(n_rows, dtype=np.float32)
    e_ew_arr = np.empty(n_rows, dtype=np.float32)
    r_ns_after_arr = np.empty(n_rows, dtype=np.float32)
    r_ew_after_arr = np.empty(n_rows, dtype=np.float32)
    n_ns_arr = np.empty(n_rows, dtype=np.int32)
    n_ew_arr = np.empty(n_rows, dtype=np.int32)
    delta_before_arr = np.empty(n_rows, dtype=np.float32)
    delta_after_arr = np.empty(n_rows, dtype=np.float32)

    # Pre-compute indices for better performance
    ns_indices = np.array([ns_index[ns] for ns in ns_pairs], dtype=np.int32)
    ew_indices = np.array([ew_index[ew] for ew in ew_pairs], dtype=np.int32)
    
    # Optimized loop with pre-computed values and reduced function calls
    for i in range(n_rows):
        idx_ns = ns_indices[i]
        idx_ew = ew_indices[i]

        r_ns = ratings_ns[idx_ns]
        r_ew = ratings_ew[idx_ew]

        # Expected probability calculation
        rating_diff = r_ns - r_ew
        e_ns = 1.0 / (1.0 + 10.0 ** (-rating_diff / elo_scale))
        e_ew = 1.0 - e_ns

        # K factor calculation with provisional boost (based on boards played)
        provisional_boost_ns = 1.5 if boards_played_ns[idx_ns] < provisional_boost_until else 1.0
        provisional_boost_ew = 1.5 if boards_played_ew[idx_ew] < provisional_boost_until else 1.0
        k_ns = k_base * provisional_boost_ns * scale[i]
        k_ew = k_base * provisional_boost_ew * scale[i]

        r_ns_before = r_ns
        r_ew_before = r_ew

        # Update ratings if valid score exists
        s_ns = pct_ns[i]
        if s_ns is not None and not (isinstance(s_ns, float) and np.isnan(s_ns)):
            # Optionally amplify score around 0.5 and clamp to [0,1]
            s_ns_f = float(s_ns)
            if score_amplifier != 1.0:
                s_ns_f = 0.5 + score_amplifier * (s_ns_f - 0.5)
                if s_ns_f < 0.0:
                    s_ns_f = 0.0
                elif s_ns_f > 1.0:
                    s_ns_f = 1.0
            s_ew_f = 1.0 - s_ns_f
            
            # Update ratings using Elo formula
            r_ns += k_ns * (s_ns_f - e_ns)
            r_ew += k_ew * (s_ew_f - e_ew)
            
            # Update arrays
            ratings_ns[idx_ns] = r_ns
            ratings_ew[idx_ew] = r_ew
            boards_played_ns[idx_ns] += 1
            boards_played_ew[idx_ew] += 1
            
            # Track unique sessions
            sessions_ns[idx_ns].add(session_ids[i])
            sessions_ew[idx_ew].add(session_ids[i])

        # Store results efficiently
        r_ns_before_arr[i] = r_ns_before
        r_ew_before_arr[i] = r_ew_before
        e_ns_arr[i] = e_ns
        e_ew_arr[i] = e_ew
        r_ns_after_arr[i] = r_ns
        r_ew_after_arr[i] = r_ew
        n_ns_arr[i] = len(sessions_ns[idx_ns])  # Count unique sessions
        n_ew_arr[i] = len(sessions_ew[idx_ew])  # Count unique sessions
        delta_before_arr[i] = rating_diff
        delta_after_arr[i] = r_ns - r_ew

    # Set Elo ratings to None if session count is less than minimum_sessions
    r_ns_final = np.where(n_ns_arr < minimum_sessions, np.nan, r_ns_after_arr)
    r_ew_final = np.where(n_ew_arr < minimum_sessions, np.nan, r_ew_after_arr)
    r_ns_before_final = np.where(n_ns_arr < minimum_sessions, np.nan, r_ns_before_arr)
    r_ew_before_final = np.where(n_ew_arr < minimum_sessions, np.nan, r_ew_before_arr)

    out_df = pl.DataFrame({
        "Elo_R_NS_Before": r_ns_before_final,
        "Elo_R_EW_Before": r_ew_before_final,
        "Elo_E_NS": e_ns_arr,
        "Elo_E_EW": e_ew_arr,
        "Elo_R_NS": r_ns_final,
        "Elo_R_EW": r_ew_final,
        "Elo_N_NS": n_ns_arr,
        "Elo_N_EW": n_ew_arr,
        "Elo_Delta_Before": delta_before_arr,
        "Elo_Delta_After": delta_after_arr,
    })
    if replace_existing:
        cols = [c for c in out_df.columns if c in df_sorted.columns]
        if cols:
            df_sorted = df_sorted.drop(cols)
    return df_sorted.hstack(out_df)


def compute_player_matchpoint_elo_ratings(
    df_sorted: pl.DataFrame,
    *,
    initial_rating: float = 1500.0,
    k_base: float = 24.0,
    provisional_boost_until: int = 100,  # boards per player-direction
    minimum_sessions: int = 10,  # minimum sessions required for non-None rating
    elo_scale: float = 400.0,
    score_amplifier: float = 1.0,
    replace_existing: bool = True,
) -> pl.DataFrame:
    """Compute player Elo-style ratings for duplicate boards.

    Purpose:
    - Calculate individual player Elo-style skill ratings based on bridge game results
    - Track rating changes for each player at each table position (N/S/E/W)
    - Provide leakage-safe updates that prevent future information contamination
    - Generate expected scores and rating progression analytics per player

    Parameters:
    - df_sorted: Sorted DataFrame ('Date','session_id','Round','Board') containing duplicate bridge game results with individual player IDs
    - initial_rating: starting rating for all players (default 1500.0)
    - k_base: base K-factor for rating updates (default 24.0)
    - provisional_boost_until: number of games before reducing K-factor (default 100)
    - minimum_sessions: minimum sessions required for non-None rating (default 10)
    - elo_scale: denominator in logistic expectation (chess default 400.0)
    - score_amplifier: amplifies per-board score around 0.5 (1.0 = no change)
    - replace_existing: if True, drop existing output columns before adding new ones.

    Input columns:
    - `Date`: Game date for temporal ordering (any comparable type)
    - `Player_ID_N`, `Player_ID_S`, `Player_ID_E`, `Player_ID_W`: Individual player identifiers
    - `Pct_NS`: North-South percentage score in [0,1] range (pl.Float32)
    - `session_id`: Session identifier (required for session counting)

    Output columns:
    - `Elo_R_N_Before`, `Elo_R_S_Before`, `Elo_R_E_Before`, `Elo_R_W_Before`: Pre-game ratings (pl.Float32, None if Elo_N < minimum_sessions)
    - `Elo_E_NS`, `Elo_E_EW`: Expected scores from pre-board partnership ratings (pl.Float32)
    - `Elo_R_N`, `Elo_R_S`, `Elo_R_E`, `Elo_R_W`: Post-game ratings (pl.Float32, None if Elo_N < minimum_sessions)  
    - `Elo_N_N`, `Elo_N_S`, `Elo_N_E`, `Elo_N_W`: Session counts per player (pl.Int32)
    
    Note: Elo ratings are set to None for players with fewer than minimum_sessions sessions played to ensure statistical reliability.

    Returns:
    - Original DataFrame with appended per-player ratings before/after, expected partnership scores, and session counts.
    """
    # TODO? (polars): Similar to pair Elo, consider chunking by session and preparing
    # arrays via Polars pre-aggregation. Full vectorization is constrained by
    # sequential updates but can reduce Python overhead with chunked updates.
    need_cols = {
        "Player_ID_N",
        "Player_ID_S",
        "Player_ID_E",
        "Player_ID_W",
        "Pct_NS",
        "session_id",
    }
    missing = need_cols - set(df_sorted.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Extract arrays
    n_rows = df_sorted.height
    pid_n_arr = df_sorted["Player_ID_N"].to_numpy()
    pid_s_arr = df_sorted["Player_ID_S"].to_numpy()
    pid_e_arr = df_sorted["Player_ID_E"].to_numpy()
    pid_w_arr = df_sorted["Player_ID_W"].to_numpy()
    pct_ns = df_sorted["Pct_NS"].to_numpy()
    session_ids = df_sorted["session_id"].to_numpy()

    # Section-size scaling vector (same precedence as pair Elo)
    scale = np.ones(n_rows, dtype=np.float32)
    if "Section_Pairs" in df_sorted.columns:
        pairs = df_sorted["Section_Pairs"].cast(pl.Float32).to_numpy()
    elif "Num_Pairs" in df_sorted.columns:
        pairs = df_sorted["Num_Pairs"].cast(pl.Float32).to_numpy()
    elif "MP_Top" in df_sorted.columns:
        pairs = (df_sorted["MP_Top"].cast(pl.Float32).to_numpy() / 2.0) + 1.0
    else:
        pairs = None
    if pairs is not None:
        with np.errstate(invalid="ignore"):
            valid = pairs > 1.0
        scale_valid = np.sqrt(np.maximum(pairs[valid] - 1.0, 1.0) / 11.0)
        scale[valid] = scale_valid

    # Map player ids to contiguous indices per direction
    ns_player_unique = pl.concat([
        df_sorted["Player_ID_N"],
        df_sorted["Player_ID_S"],
    ]).unique().to_list()
    ew_player_unique = pl.concat([
        df_sorted["Player_ID_E"],
        df_sorted["Player_ID_W"],
    ]).unique().to_list()

    idx_ns_map = {pid: i for i, pid in enumerate(ns_player_unique)}
    idx_ew_map = {pid: i for i, pid in enumerate(ew_player_unique)}

    ratings_ns = np.full(len(ns_player_unique), initial_rating, dtype=np.float32)
    ratings_ew = np.full(len(ew_player_unique), initial_rating, dtype=np.float32)
    boards_played_ns = np.zeros(len(ns_player_unique), dtype=np.int32)  # Track boards for provisional boost
    boards_played_ew = np.zeros(len(ew_player_unique), dtype=np.int32)  # Track boards for provisional boost
    
    # Track unique sessions per player using sets
    sessions_ns = [set() for _ in range(len(ns_player_unique))]
    sessions_ew = [set() for _ in range(len(ew_player_unique))]

    # Outputs
    R_N_Before = np.empty(n_rows, dtype=np.float32)
    R_S_Before = np.empty(n_rows, dtype=np.float32)
    R_E_Before = np.empty(n_rows, dtype=np.float32)
    R_W_Before = np.empty(n_rows, dtype=np.float32)
    NS_side_Before = np.empty(n_rows, dtype=np.float32)
    EW_side_Before = np.empty(n_rows, dtype=np.float32)
    E_NS = np.empty(n_rows, dtype=np.float32)
    E_EW = np.empty(n_rows, dtype=np.float32)
    R_N_after = np.empty(n_rows, dtype=np.float32)
    R_S_after = np.empty(n_rows, dtype=np.float32)
    R_E_after = np.empty(n_rows, dtype=np.float32)
    R_W_after = np.empty(n_rows, dtype=np.float32)
    N_N = np.empty(n_rows, dtype=np.int32)
    N_S = np.empty(n_rows, dtype=np.int32)
    N_E = np.empty(n_rows, dtype=np.int32)
    N_W = np.empty(n_rows, dtype=np.int32)
    Elo_Delta_Before = np.empty(n_rows, dtype=np.float32)

    # Pre-compute all indices for better performance
    idx_n_arr = np.array([idx_ns_map[pid] for pid in pid_n_arr], dtype=np.int32)
    idx_s_arr = np.array([idx_ns_map[pid] for pid in pid_s_arr], dtype=np.int32)
    idx_e_arr = np.array([idx_ew_map[pid] for pid in pid_e_arr], dtype=np.int32)
    idx_w_arr = np.array([idx_ew_map[pid] for pid in pid_w_arr], dtype=np.int32)
    
    # Optimized loop with pre-computed indices
    for i in range(n_rows):
        idx_n = idx_n_arr[i]
        idx_s = idx_s_arr[i]
        idx_e = idx_e_arr[i]
        idx_w = idx_w_arr[i]

        r_n_ns = ratings_ns[idx_n]
        r_s_ns = ratings_ns[idx_s]
        r_e_ew = ratings_ew[idx_e]
        r_w_ew = ratings_ew[idx_w]

        ns_before = (r_n_ns + r_s_ns) / 2.0
        ew_before = (r_e_ew + r_w_ew) / 2.0

        e_ns = 1.0 / (1.0 + 10.0 ** (-(ns_before - ew_before) / elo_scale))
        e_ew = 1.0 - e_ns

        # K per player with provisional boost and section-size scale (based on boards played)
        k_n = k_base * (1.5 if boards_played_ns[idx_n] < provisional_boost_until else 1.0) * scale[i]
        k_s = k_base * (1.5 if boards_played_ns[idx_s] < provisional_boost_until else 1.0) * scale[i]
        k_e = k_base * (1.5 if boards_played_ew[idx_e] < provisional_boost_until else 1.0) * scale[i]
        k_w = k_base * (1.5 if boards_played_ew[idx_w] < provisional_boost_until else 1.0) * scale[i]

        r_n_before = r_n_ns
        r_s_before = r_s_ns
        r_e_before = r_e_ew
        r_w_before = r_w_ew

        s_ns = pct_ns[i]
        if s_ns is not None and not (isinstance(s_ns, float) and np.isnan(s_ns)):
            # Calculate proper deltas for each direction
            s_ns_f = float(s_ns)
            if score_amplifier != 1.0:
                s_ns_f = 0.5 + score_amplifier * (s_ns_f - 0.5)
                if s_ns_f < 0.0:
                    s_ns_f = 0.0
                elif s_ns_f > 1.0:
                    s_ns_f = 1.0
            delta_ns = s_ns_f - e_ns
            delta_ew = (1.0 - s_ns_f) - e_ew  # Use EW score for EW players
            r_n_ns = r_n_ns + 0.5 * k_n * delta_ns
            r_s_ns = r_s_ns + 0.5 * k_s * delta_ns
            r_e_ew = r_e_ew + 0.5 * k_e * delta_ew  # Use EW delta, not NS delta
            r_w_ew = r_w_ew + 0.5 * k_w * delta_ew  # Use EW delta, not NS delta
            ratings_ns[idx_n] = r_n_ns
            ratings_ns[idx_s] = r_s_ns
            ratings_ew[idx_e] = r_e_ew
            ratings_ew[idx_w] = r_w_ew
            boards_played_ns[idx_n] += 1
            boards_played_ns[idx_s] += 1
            boards_played_ew[idx_e] += 1
            boards_played_ew[idx_w] += 1
            
            # Track unique sessions
            sessions_ns[idx_n].add(session_ids[i])
            sessions_ns[idx_s].add(session_ids[i])
            sessions_ew[idx_e].add(session_ids[i])
            sessions_ew[idx_w].add(session_ids[i])

        # Save outputs
        R_N_Before[i] = r_n_before
        R_S_Before[i] = r_s_before
        R_E_Before[i] = r_e_before
        R_W_Before[i] = r_w_before
        NS_side_Before[i] = ns_before
        EW_side_Before[i] = ew_before
        E_NS[i] = e_ns
        E_EW[i] = e_ew
        R_N_after[i] = r_n_ns
        R_S_after[i] = r_s_ns
        R_E_after[i] = r_e_ew
        R_W_after[i] = r_w_ew
        N_N[i] = len(sessions_ns[idx_n])  # Count unique sessions
        N_S[i] = len(sessions_ns[idx_s])  # Count unique sessions
        N_E[i] = len(sessions_ew[idx_e])  # Count unique sessions
        N_W[i] = len(sessions_ew[idx_w])  # Count unique sessions
        Elo_Delta_Before[i] = ns_before - ew_before

    # Set Elo ratings to None if session count is less than minimum_sessions
    R_N_final = np.where(N_N < minimum_sessions, np.nan, R_N_after)
    R_S_final = np.where(N_S < minimum_sessions, np.nan, R_S_after)
    R_E_final = np.where(N_E < minimum_sessions, np.nan, R_E_after)
    R_W_final = np.where(N_W < minimum_sessions, np.nan, R_W_after)
    R_N_before_final = np.where(N_N < minimum_sessions, np.nan, R_N_Before)
    R_S_before_final = np.where(N_S < minimum_sessions, np.nan, R_S_Before)
    R_E_before_final = np.where(N_E < minimum_sessions, np.nan, R_E_Before)
    R_W_before_final = np.where(N_W < minimum_sessions, np.nan, R_W_Before)

    out_df = pl.DataFrame({
        "Elo_R_N_Before": R_N_before_final,
        "Elo_R_E_Before": R_E_before_final,
        "Elo_R_S_Before": R_S_before_final,
        "Elo_R_W_Before": R_W_before_final,
        "Elo_E_NS": E_NS,
        "Elo_E_EW": E_EW,
        "Elo_R_N": R_N_final,
        "Elo_R_E": R_E_final,
        "Elo_R_S": R_S_final,
        "Elo_R_W": R_W_final,
        "Elo_N_N": N_N,
        "Elo_N_E": N_E,
        "Elo_N_S": N_S,
        "Elo_N_W": N_W,
    })
    if replace_existing:
        cols = [c for c in out_df.columns if c in df_sorted.columns]
        if cols:
            df_sorted = df_sorted.drop(cols)
    return df_sorted.hstack(out_df)
def compute_event_start_end_elo_columns(df_sorted: pl.DataFrame) -> pl.DataFrame:
    """Add constant per-session Elo columns for each seat and pair.

    Purpose:
    - Calculate event-level Elo ratings for players and pairs at session start and end
    - Propagate constant rating values across all boards within each session
    - Support session-level rating analysis and progression tracking
    - Enable before/after event rating comparisons for performance assessment

    Parameters:
    - df_sorted: Sorted DataFrame ('Date','session_id','Round','Board') containing board-level Elo rating data

    Input columns:
    - `Elo_R_{N|S|E|W}_Before`: Pre-board player ratings
    - `Elo_R_{N|S|E|W}_After`: Post-board player ratings  
    - `Elo_R_{NS|EW}_Before`: Pre-board pair ratings
    - `Elo_R_{NS|EW}_After`: Post-board pair ratings
    - `session_id`: session identifier

    Output columns:
    - `Elo_R_{N|S|E|W}_EventStart`: Player rating at session start (pl.Float32)
    - `Elo_R_{N|S|E|W}_EventEnd`: Player rating at session end (pl.Float32)
    - `Elo_R_{NS|EW}_EventStart`: Pair rating at session start (pl.Float32)
    - `Elo_R_{NS|EW}_EventEnd`: Pair rating at session end (pl.Float32)

    Returns:
    - DataFrame with added event-level Elo columns propagated across all session boards
    """
    t = time.time()

    # Player start
    seat_before_cols = {
        "N": ("Player_ID_N", "Elo_R_N_Before", "Elo_R_N_EventStart"),
        "S": ("Player_ID_S", "Elo_R_S_Before", "Elo_R_S_EventStart"),
        "E": ("Player_ID_E", "Elo_R_E_Before", "Elo_R_E_EventStart"),
        "W": ("Player_ID_W", "Elo_R_W_Before", "Elo_R_W_EventStart"),
    }
    for _, (pid_col, before_col, out_col) in seat_before_cols.items():
        if pid_col in df_sorted.columns and before_col in df_sorted.columns:
            df_sorted = df_sorted.with_columns(
                pl.col(before_col).first().over(["session_id", pid_col]).alias(out_col)
            )

    # Pair start
    pair_before_cols = {
        "NS": ("Pair_Number_NS", "Elo_R_NS_Before", "Elo_R_NS_EventStart"),
        "EW": ("Pair_Number_EW", "Elo_R_EW_Before", "Elo_R_EW_EventStart"),
    }
    for _, (pair_col, before_col, out_col) in pair_before_cols.items():
        if pair_col in df_sorted.columns and before_col in df_sorted.columns:
            df_sorted = df_sorted.with_columns(
                pl.col(before_col).first().over(["session_id", pair_col]).alias(out_col)
            )

    # Player end
    seat_after_cols = {
        "N": ("Player_ID_N", "Elo_R_N", "Elo_R_N_EventEnd"),
        "S": ("Player_ID_S", "Elo_R_S", "Elo_R_S_EventEnd"),
        "E": ("Player_ID_E", "Elo_R_E", "Elo_R_E_EventEnd"),
        "W": ("Player_ID_W", "Elo_R_W", "Elo_R_W_EventEnd"),
    }
    for _, (pid_col, after_col, out_col) in seat_after_cols.items():
        if pid_col in df_sorted.columns and after_col in df_sorted.columns:
            df_sorted = df_sorted.with_columns(
                pl.col(after_col).last().over(["session_id", pid_col]).alias(out_col)
            )

    # Pair end
    pair_after_cols = {
        "NS": ("Pair_Number_NS", "Elo_R_NS", "Elo_R_NS_EventEnd"),
        "EW": ("Pair_Number_EW", "Elo_R_EW", "Elo_R_EW_EventEnd"),
    }
    for _, (pair_col, after_col, out_col) in pair_after_cols.items():
        if pair_col in df_sorted.columns and after_col in df_sorted.columns:
            df_sorted = df_sorted.with_columns(
                pl.col(after_col).last().over(["session_id", pair_col]).alias(out_col)
            )

    logger.info(f"Event start/end Elo columns: {time.time()-t:.3f} seconds")
    return df_sorted


def compute_matchpoint_elo_ratings(
    df: pl.DataFrame,
    *,
    initial_rating: float = 1500.0,
    k_base: float = 24.0,
    provisional_boost_until: int = 100,
    minimum_sessions: int = 10,
    elo_scale: float = 400.0,
    score_amplifier: float = 1.0,
) -> pl.DataFrame:
    """Compute matchpoint Elo ratings for players and pairs.

    Purpose:
    - Calculate Elo ratings for players and pairs based on matchpoint results
    - Propagate Elo ratings across all boards within each session
    - Support session-level rating analysis and progression tracking
    - Enable before/after event rating comparisons for performance assessment

    Parameters:
    - df: Polars DataFrame containing board-level Elo rating data
    - initial_rating: Starting Elo rating for all entities (default 1500.0)
    - k_base: Base K-factor for rating adjustments (default 24.0)
    - provisional_boost_until: Number of boards before reducing K-factor (default 100)
    - minimum_sessions: Minimum sessions required for non-None rating (default 10)
    - elo_scale: Elo scale parameter for expected score calculation (default 400.0)
    - score_amplifier: Multiplier for score differences around 0.5 (default 1.0)

    Input columns:
    - `session_id`: session identifier
    - `Pct_NS`: percentage of matchpoints won by North-South
    - `MP_Top`: matchpoint top

    Output columns:
    
    From compute_pair_matchpoint_elo_ratings():
    - `Elo_R_NS_Before`, `Elo_R_EW_Before`: Pair ratings before the board (pl.Float32, None if < minimum_sessions)
    - `Elo_E_NS`, `Elo_E_EW`: Expected scores based on pair rating difference (pl.Float32)
    - `Elo_R_NS`, `Elo_R_EW`: Pair ratings after the board (pl.Float32, None if < minimum_sessions)
    - `Elo_N_NS`, `Elo_N_EW`: Number of sessions played by each pair (pl.Int32)
    - `Elo_Delta_Before`, `Elo_Delta_After`: Rating differences NS minus EW (pl.Float32)
    
    From compute_player_matchpoint_elo_ratings():
    - `Elo_R_N_Before`, `Elo_R_S_Before`, `Elo_R_E_Before`, `Elo_R_W_Before`: Player ratings before the board (pl.Float32, None if < minimum_sessions)
    - `Elo_E_NS`, `Elo_E_EW`: Expected scores from partnership ratings (pl.Float32)
    - `Elo_R_N`, `Elo_R_S`, `Elo_R_E`, `Elo_R_W`: Player ratings after the board (pl.Float32, None if < minimum_sessions)
    - `Elo_N_N`, `Elo_N_S`, `Elo_N_E`, `Elo_N_W`: Number of sessions played by each player (pl.Int32)
    
    From compute_event_start_end_elo_columns():
    - `Elo_R_N_EventStart`, `Elo_R_S_EventStart`, `Elo_R_E_EventStart`, `Elo_R_W_EventStart`: Player ratings at session start (pl.Float32)
    - `Elo_R_N_EventEnd`, `Elo_R_S_EventEnd`, `Elo_R_E_EventEnd`, `Elo_R_W_EventEnd`: Player ratings at session end (pl.Float32)
    - `Elo_R_NS_EventStart`, `Elo_R_EW_EventStart`: Pair ratings at session start (pl.Float32)
    - `Elo_R_NS_EventEnd`, `Elo_R_EW_EventEnd`: Pair ratings at session end (pl.Float32)

    Returns:
    - Polars DataFrame with added matchpoint Elo rating columns (36 total columns: 10 pair + 14 player + 12 event-level)
    """

    if 'MP_Top' not in df.columns:
        df = add_board_matchpoint_top(df)

    if 'Pct_NS' not in df.columns:
        df = add_percentage_scores(df)

    # takes 2m30s. Adds 10 columns.
    df = compute_pair_matchpoint_elo_ratings(
        df,
        initial_rating=initial_rating,
        k_base=k_base,
        provisional_boost_until=provisional_boost_until,
        minimum_sessions=minimum_sessions,
        elo_scale=elo_scale,
        score_amplifier=score_amplifier,
    )

    # takes 6m. Adds 14 columns.
    df = compute_player_matchpoint_elo_ratings(
        df, # select minimal columns
        initial_rating=initial_rating,
        k_base=k_base,
        provisional_boost_until=provisional_boost_until,
        minimum_sessions=minimum_sessions,
        elo_scale=elo_scale,
        score_amplifier=score_amplifier,
    )

    # takes 30s. Adds 12 columns.
    df = compute_event_start_end_elo_columns(
        df, # select minimal columns
    )

    return df


def add_contract_types(df: pl.DataFrame) -> pl.DataFrame:
    """Create contract type columns CT_[NESW]_[SHDCN] from DD tricks.

    Purpose:
    - Derive categorical contract types (Pass, Partial, Game, SSlam, GSlam) per seat and strain.

    Parameters:
    - df: input DataFrame.

    Input columns:
    - `DD_[NESW]_[SHDCN]` trick counts.

    Output columns:
    - `CT_[NESW]_[SHDCN]` strings.

    Returns:
    - DataFrame with CT columns (no-op if already present).
    """
    if 'CT_N_C' in df.columns:
        return df
    ct_columns = [
        pl.when(pl.col(f"DD_{direction}_{strain}") < 7).then(pl.lit("Pass"))
        .when((pl.col(f"DD_{direction}_{strain}") == 11) & (strain in ['C', 'D'])).then(pl.lit("Game"))
        .when((pl.col(f"DD_{direction}_{strain}").is_in([10, 11])) & (strain in ['H', 'S'])).then(pl.lit("Game"))
        .when((pl.col(f"DD_{direction}_{strain}").is_in([9, 10, 11])) & (strain == 'N')).then(pl.lit("Game"))
        .when(pl.col(f"DD_{direction}_{strain}") == 12).then(pl.lit("SSlam"))
        .when(pl.col(f"DD_{direction}_{strain}") == 13).then(pl.lit("GSlam"))
        .otherwise(pl.lit("Partial")).alias(f"CT_{direction}_{strain}")
        for direction in "NESW" for strain in "SHDCN"
    ]
    return df.with_columns(ct_columns)


def add_contract_type_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Create boolean CT columns from CT types for seats and pairs.

    Purpose:
    - Convert categorical contract types to boolean flag columns for analysis
    - Enable filtering and aggregation by specific contract types
    - Create binary indicators for each contract category by seat and pair
    - Support statistical analysis of contract type distributions

    Parameters:
    - df: DataFrame containing contract type categorical information

    Input columns:
    - `CT_[NESW]_[SHDCN]`: Contract type categories by seat and strain

    Output columns:
    - `CT_[NESW]_[SHDCN]_{Pass|Partial|Game|SSlam|GSlam}`: Boolean flags by seat (pl.Boolean)
    - `CT_{NS|EW}_[SHDCN]_{Pass|Partial|Game|SSlam|GSlam}`: Boolean flags by pair (pl.Boolean)

    Returns:
    - DataFrame with added contract type boolean flag columns
    """
    out_df = df
    if 'CT_N_C_Game' not in out_df.columns:
        ct_boolean_columns = [
            pl.col(f"CT_{direction}_{strain}").eq(pl.lit(contract_type)).alias(
                f"CT_{direction}_{strain}_{contract_type}"
            )
            for direction in "NESW" for strain in "SHDCN"
            for contract_type in ["Pass", "Game", "SSlam", "GSlam", "Partial"]
        ]
        out_df = out_df.with_columns(ct_boolean_columns)

    if 'CT_NS_C_Game' not in out_df.columns:
        ct_pair_boolean_columns = [
            (
                pl.col(f"CT_{pair[0]}_{strain}_{contract_type}")
                | pl.col(f"CT_{pair[1]}_{strain}_{contract_type}")
            ).alias(f"CT_{pair}_{strain}_{contract_type}")
            for pair in ["NS", "EW"] for strain in "SHDCN"
            for contract_type in ["Pass", "Game", "SSlam", "GSlam", "Partial"]
        ]
        out_df = out_df.with_columns(ct_pair_boolean_columns)
    return out_df


def normalize_contract_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize contract-related columns and derive direction helpers.

    Purpose:
    - Extract and standardize contract information from raw contract strings
    - Generate derived positional columns for bridge analysis
    - Create vulnerability flags specific to the declaring partnership
    - Establish directional references (LHO, dummy, RHO) for play analysis

    Parameters:
    - df: DataFrame containing contract and player information

    Input columns:
    - `Contract`: Raw contract string or parts derivable from contract
    - `Player_Name_*`: Player name columns for each direction

    Output columns:
    - `Contract`: Standardized contract string (pl.String)
    - `BidLvl`: Bid level (1-7) (pl.UInt8)
    - `BidSuit`: Bid suit (S/H/D/C/N) (pl.String)
    - `Dbl`: Doubling indicator (pl.String)
    - `Declarer_Direction`: Declarer direction (N/E/S/W) (pl.String)
    - `Pair_Declarer`: Declaring partnership (NS/EW) (pl.String)
    - `Vul_Declarer`: Declarer vulnerability flag (pl.Boolean)
    - `LHO_Direction`, `Dummy_Direction`, `RHO_Direction`: Positional references (pl.String)

    Returns:
    - DataFrame with normalized contract columns and derived directional helpers
    """
    assert 'Player_Name_N' in df.columns
    df = df.with_columns(
        pl.Series('Contract', standardize_contract_format(df), pl.String, strict=False)
    )
    df = extract_contract_doubling(
        extract_contract_strain(
            extract_contract_level(
                extract_vulnerable_declarer(
                    extract_declarer_pair(
                        extract_declarer_from_contract(df)
                    )
                )
            )
        )
    )
    if ('LHO_Direction' not in df.columns and
        'Dummy_Direction' not in df.columns and
        'RHO_Direction' not in df.columns):
        df = df.with_columns([
            pl.col('Declarer_Direction').replace_strict(declarer_to_LHO_d).alias('LHO_Direction'),
            pl.col('Declarer_Direction').replace_strict(declarer_to_dummy_d).alias('Dummy_Direction'),
            pl.col('Declarer_Direction').replace_strict(declarer_to_RHO_d).alias('RHO_Direction'),
        ])
    return df


def build_vulnerability_conditions() -> Dict[str, pl.Expr]:
    """Create vulnerability condition expressions for pair-specific vulnerability logic.

    Purpose:
    - Generate Polars expressions for vulnerability-based conditional logic
    - Enable efficient vulnerability checks in complex calculations
    - Provide reusable vulnerability conditions for multiple operations

    Parameters:
    - df: DataFrame containing vulnerability information (not directly used)

    Returns:
    - Dict[str, pl.Expr]: Dictionary mapping pair names to vulnerability expressions

    Input columns:
    - 'Vul': String vulnerability column referenced in returned expressions

    Output columns:
    - Returns expressions, not DataFrame columns directly

    Returns:
    - 'NS': Expression checking if North-South is vulnerable
    - 'EW': Expression checking if East-West is vulnerable
    """
    return {
        'NS': pl.col('Vul').is_in(['N_S', 'Both']),
        'EW': pl.col('Vul').is_in(['E_W', 'Both']),
    }


def build_ev_expressions_for_pair(pd: str, get_vulnerability_conditions: Dict[str, pl.Expr]) -> list[pl.Expr]:
    """Build expected value expressions for a pair considering vulnerability.

    Purpose:
    - Create vulnerability-conditional expressions for expected value calculations
    - Select appropriate vulnerable/non-vulnerable EV values based on board conditions
    - Generate both max value and max column reference expressions

    Parameters:
    - pd: Pair direction ('NS' or 'EW')
    - get_vulnerability_conditions: Dictionary mapping pairs to vulnerability expressions

    Returns:
    - list[pl.Expr]: List of Polars expressions for vulnerability-conditional EV

    Input columns:
    - 'EV_{pd}_V_Max': Maximum vulnerable EV for the pair
    - 'EV_{pd}_NV_Max': Maximum non-vulnerable EV for the pair
    - 'EV_{pd}_V_Max_Col': Column name with maximum vulnerable EV
    - 'EV_{pd}_NV_Max_Col': Column name with maximum non-vulnerable EV

    Output columns:
    - 'EV_Max_{pd}': Vulnerability-conditional maximum EV
    - 'EV_Max_Col_{pd}': Vulnerability-conditional max EV column reference
    """
    return [
        pl.when(get_vulnerability_conditions[pd]).then(pl.col(f'EV_{pd}_V_Max')).otherwise(pl.col(f'EV_{pd}_NV_Max')).alias(f'EV_Max_{pd}'),
        pl.when(get_vulnerability_conditions[pd]).then(pl.col(f'EV_{pd}_V_Max_Col')).otherwise(pl.col(f'EV_{pd}_NV_Max_Col')).alias(f'EV_Max_Col_{pd}')
    ]


def build_ev_expressions_for_declarer(pd: str, dd: str, get_vulnerability_conditions: Dict[str, pl.Expr]) -> list[pl.Expr]:
    """Build expected value expressions for specific declarer position considering vulnerability.

    Purpose:
    - Create vulnerability-conditional expressions for declarer-specific EV calculations
    - Select appropriate vulnerable/non-vulnerable EV values for specific declarers
    - Generate both max value and max column reference expressions

    Parameters:
    - pd: Pair direction ('NS' or 'EW')
    - dd: Declarer direction ('N', 'E', 'S', or 'W')
    - get_vulnerability_conditions: Dictionary mapping pairs to vulnerability expressions

    Returns:
    - list[pl.Expr]: List of Polars expressions for vulnerability-conditional declarer EV

    Input columns:
    - 'EV_{pd}_{dd}_V_Max': Maximum vulnerable EV for specific declarer
    - 'EV_{pd}_{dd}_NV_Max': Maximum non-vulnerable EV for specific declarer
    - 'EV_{pd}_{dd}_V_Max_Col': Column name with maximum vulnerable EV for declarer
    - 'EV_{pd}_{dd}_NV_Max_Col': Column name with maximum non-vulnerable EV for declarer

    Output columns:
    - 'EV_{pd}_{dd}_Max': Vulnerability-conditional maximum EV for declarer
    - 'EV_{pd}_{dd}_Max_Col': Vulnerability-conditional max EV column reference for declarer
    """
    return [
        pl.when(get_vulnerability_conditions[pd]).then(pl.col(f'EV_{pd}_{dd}_V_Max')).otherwise(pl.col(f'EV_{pd}_{dd}_NV_Max')).alias(f'EV_{pd}_{dd}_Max'),
        pl.when(get_vulnerability_conditions[pd]).then(pl.col(f'EV_{pd}_{dd}_V_Max_Col')).otherwise(pl.col(f'EV_{pd}_{dd}_NV_Max_Col')).alias(f'EV_{pd}_{dd}_Max_Col')
    ]


def build_ev_expressions_for_strain(pd: str, dd: str, s: str, get_vulnerability_conditions: Dict[str, pl.Expr]) -> list[pl.Expr]:
    """Build expected value expressions for specific strain considering vulnerability.

    Purpose:
    - Create vulnerability-conditional expressions for strain-specific EV calculations
    - Select appropriate vulnerable/non-vulnerable EV values for specific strains
    - Generate max value and column reference expressions

    Parameters:
    - pd: Pair direction ('NS' or 'EW')
    - dd: Declarer direction ('N', 'E', 'S', or 'W')
    - s: Strain/suit ('S', 'H', 'D', 'C', or 'N')
    - get_vulnerability_conditions: Dictionary mapping pairs to vulnerability expressions

    Returns:
    - list[pl.Expr]: List of Polars expressions for vulnerability-conditional strain EV

    Input columns:
    - 'EV_{pd}_{dd}_{s}_V_Max': Maximum vulnerable EV for specific strain
    - 'EV_{pd}_{dd}_{s}_NV_Max': Maximum non-vulnerable EV for specific strain
    - 'EV_{pd}_{dd}_{s}_V_Max_Col': Column name with maximum vulnerable EV for strain
    - 'EV_{pd}_{dd}_{s}_NV_Max_Col': Column name with maximum non-vulnerable EV for strain

    Output columns:
    - 'EV_{pd}_{dd}_{s}_Max': Vulnerability-conditional maximum EV for strain
    - 'EV_{pd}_{dd}_{s}_Max_Col': Vulnerability-conditional max EV column reference
    """
    return [
        pl.when(get_vulnerability_conditions[pd]).then(pl.col(f'EV_{pd}_{dd}_{s}_V_Max')).otherwise(pl.col(f'EV_{pd}_{dd}_{s}_NV_Max')).alias(f'EV_{pd}_{dd}_{s}_Max'),
        pl.when(get_vulnerability_conditions[pd]).then(pl.col(f'EV_{pd}_{dd}_{s}_V_Max_Col')).otherwise(pl.col(f'EV_{pd}_{dd}_{s}_NV_Max_Col')).alias(f'EV_{pd}_{dd}_{s}_Max_Col'),
    ]


def build_ev_expressions_for_level(pd: str, dd: str, s: str, l: int, get_vulnerability_conditions: Dict[str, pl.Expr]) -> list[pl.Expr]:
    """Build expected value expressions for specific contract level considering vulnerability.

    Purpose:
    - Create vulnerability-conditional expressions for level-specific EV calculations
    - Select appropriate vulnerable/non-vulnerable EV values for specific contract levels
    - Generate final contract-specific EV expressions

    Parameters:
    - pd: Pair direction ('NS' or 'EW')
    - dd: Declarer direction ('N', 'E', 'S', or 'W')
    - s: Strain/suit ('S', 'H', 'D', 'C', or 'N')
    - l: Contract level (1-7)
    - get_vulnerability_conditions: Dictionary mapping pairs to vulnerability expressions

    Returns:
    - list[pl.Expr]: List of Polars expressions for vulnerability-conditional level EV

    Input columns:
    - 'EV_{pd}_{dd}_{s}_{l}_V': Vulnerable EV for specific contract level
    - 'EV_{pd}_{dd}_{s}_{l}_NV': Non-vulnerable EV for specific contract level

    Output columns:
    - 'EV_{pd}_{dd}_{s}_{l}': Vulnerability-conditional EV for specific contract
    """
    return [
        pl.when(get_vulnerability_conditions[pd]).then(pl.col(f'EV_{pd}_{dd}_{s}_{l}_V')).otherwise(pl.col(f'EV_{pd}_{dd}_{s}_{l}_NV')).alias(f'EV_{pd}_{dd}_{s}_{l}')
    ]


def build_pair_ev_expressions(pd: str, get_vulnerability_conditions: Dict[str, pl.Expr]) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    expressions.extend(build_ev_expressions_for_pair(pd, get_vulnerability_conditions))
    for dd in pd:
        expressions.extend(build_ev_expressions_for_declarer(pd, dd, get_vulnerability_conditions))
        for s in 'SHDCN':
            expressions.extend(build_ev_expressions_for_strain(pd, dd, s, get_vulnerability_conditions))
            for l in range(1, 8):
                expressions.extend(build_ev_expressions_for_level(pd, dd, s, l, get_vulnerability_conditions))
    return expressions


def add_best_contract_ev(df: pl.DataFrame) -> pl.DataFrame:
    """Create EV summary columns and declarer-specific EV columns from granular EVs.

    Purpose:
    - Identify optimal contracts based on expected value calculations
    - Generate maximum EV values for each partnership
    - Create column references to identify which contracts yield maximum EV
    - Support contract selection and bidding analysis

    Parameters:
    - df: DataFrame containing comprehensive expected value calculations

    Input columns:
    - `EV_*`: Expected value grid by pair/declarer/strain/level including V/NV variants
    - `Declarer_Pair_Direction`: Partnership of the declaring side

    Output columns:
    - `EV_Max_NS`: Maximum expected value for North-South (pl.Float32)
    - `EV_Max_EW`: Maximum expected value for East-West (pl.Float32)
    - `EV_Max`: Overall maximum expected value (pl.Float32)
    - `EV_Max_Col`: Column name containing the maximum EV (pl.String)
    - `EV_Max_Declarer`: Maximum EV from declarer's perspective (pl.Float32)
    - `EV_Max_Col_Declarer`: Column name containing declarer's maximum EV (pl.String)

    Returns:
    - DataFrame with added expected value summary and identification columns
    """
    vul = build_vulnerability_conditions()
    max_expressions: list[pl.Expr] = []
    for pd in ['NS', 'EW']:
        max_expressions.extend(build_pair_ev_expressions(pd, vul))
    # TODO(polars): Consider switching to a LazyFrame here if this is part of a larger
    # pipeline to reduce intermediate materialization: df.lazy().with_columns(...).collect()
    out = df.with_columns(max_expressions)
    ev_columns = f'^EV_Max_(NS|EW)$'
    max_expr, col_expr = find_max_horizontal_value(out, ev_columns)
    out = out.with_columns([
        max_expr.alias('EV_Max'),
        col_expr.alias('EV_Max_Col'),
    ])
    out = out.with_columns([
        pl.when(pl.col('Declarer_Pair_Direction').eq('NS')).then(pl.col('EV_Max_NS')).otherwise(pl.col('EV_Max_EW')).alias('EV_Max_Declarer'),
        pl.when(pl.col('Declarer_Pair_Direction').eq('NS')).then(pl.col('EV_Max_Col_NS')).otherwise(pl.col('EV_Max_Col_EW')).alias('EV_Max_Col_Declarer'),
    ])
    return out


def add_position_role_info(df: pl.DataFrame) -> pl.DataFrame:
    """Create declarer/opponent/lead position columns and derived scores.

    Purpose:
    - Generate positional role assignments for all players at the table
    - Create player-specific identifiers for declarer, dummy, and defenders
    - Calculate role-specific scores and performance metrics
    - Support position-based analysis and player tracking

    Parameters:
    - df: DataFrame containing contract, player, and scoring information

    Input columns:
    - `Declarer_Direction`: Direction of the declaring player (N/E/S/W)
    - `Player_ID_*`: Player identifiers for each table position
    - `Declarer_Pair_Direction`: Partnership of the declarer (NS/EW)
    - `BidSuit`, `BidLvl`, `Contract`: Contract specification
    - `Score_NS`, `Score_EW`: Partnership scores
    - `Par_NS`, `Par_EW`: Theoretical par scores

    Output columns:
    - `Declarer`: Declarer player identifier
    - `Direction_OnLead`: Direction of opening leader
    - `EV_Score_Col_Declarer`: Expected value column reference for declarer
    - `Score_Declarer`, `Par_Declarer`: Declarer-specific scores
    - `Defender_Pair_Direction`: Defending partnership (non-declarer pair)
    - `Direction_Dummy`, `Dummy`: Dummy position and player identifier
    - `OnLead`, `Direction_NotOnLead`, `NotOnLead`: Lead and defensive position assignments
    - `Defender_Par_GE`: Defensive par performance indicator

    Returns:
    - DataFrame with added positional role and derived scoring columns
    """
    return (
        df.with_columns([
            pl.struct(['Declarer_Direction', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                lambda r: None if r['Declarer_Direction'] is None else r[f"Player_ID_{r['Declarer_Direction']}"] , return_dtype=pl.String
            ).alias('Declarer')
        ])
        .with_columns([
            pl.col('Declarer_Direction').replace_strict(NextPosition).alias('Direction_OnLead'),
        ])
        .with_columns([
            pl.concat_str([pl.lit('EV'), pl.col('Declarer_Pair_Direction'), pl.col('Declarer_Direction'), pl.col('BidSuit'), pl.col('BidLvl').cast(pl.String), pl.when(pl.col('Vul_Declarer')).then(pl.lit('V')).otherwise(pl.lit('NV'))], separator='_').alias('EV_Score_Col_Declarer'),
            pl.struct(['Contract','Declarer_Pair_Direction', 'Score_NS', 'Score_EW']).map_elements(
                lambda r: 0 if r['Contract'] == 'PASS' else (None if r['Declarer_Pair_Direction'] is None else r[f"Score_{r['Declarer_Pair_Direction']}"]), return_dtype=pl.Int16
            ).alias('Score_Declarer'),
            pl.when(pl.col('Declarer_Pair_Direction').eq(pl.lit('NS'))).then(pl.col('Par_NS')).otherwise(pl.col('Par_EW')).alias('Par_Declarer'),
        ])
        # Add columns for par achievements
        # Is_Par_Suit: True when the declarer achieved par for the best suit
        # Is_Par_Contract: True when the declarer achieved par for their actual contract
        # Is_Sacrifice: True when the declarer achieved par for their actual contract
        .with_columns([
            pl.col('Par_Declarer').eq(pl.col('DD_Score_Max_Declarer')).alias('Is_Par_Suit'),
            pl.col('Par_Declarer').eq(pl.col('DD_Score_Declarer')).alias('Is_Par_Contract'),
            (pl.col('Par_Declarer').eq(pl.col('DD_Score_Declarer')) & pl.col('DD_Score_Declarer').lt(0)).alias('Is_Sacrifice')
        ])
        .with_columns([
            pl.col('Declarer_Pair_Direction').replace_strict(PairDirectionToOpponentPairDirection).alias('Defender_Pair_Direction'),
            pl.col('Direction_OnLead').replace_strict(NextPosition).alias('Direction_Dummy'),
            pl.struct(['Direction_OnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                lambda r: None if r['Direction_OnLead'] is None else r[f"Player_ID_{r['Direction_OnLead']}"] , return_dtype=pl.String
            ).alias('OnLead'),
        ])
        .with_columns([
            pl.col('Direction_Dummy').replace_strict(NextPosition).alias('Direction_NotOnLead'),
            pl.struct(['Direction_Dummy', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                lambda r: None if r['Direction_Dummy'] is None else r[f"Player_ID_{r['Direction_Dummy']}"] , return_dtype=pl.String
            ).alias('Dummy'),
            pl.col('Score_Declarer').le(pl.col('Par_Declarer')).alias('Defender_Par_GE')
        ])
        .with_columns([
            pl.struct(['Direction_NotOnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                lambda r: None if r['Direction_NotOnLead'] is None else r[f"Player_ID_{r['Direction_NotOnLead']}"] , return_dtype=pl.String
            ).alias('NotOnLead')
        ])
    )
def add_trick_probabilities(df: pl.DataFrame) -> pl.DataFrame:
    """Create Prob_Taking_0..13 columns by selecting the probability matching the declarer context.

    Purpose:
    - Extract contract-specific trick-taking probabilities for the declaring side
    - Map detailed probability matrices to specific contract contexts
    - Generate probability distributions for single-dummy analysis
    - Support expected value calculations and contract evaluation

    Parameters:
    - df: DataFrame containing declarer context and probability data

    Input columns:
    - `Declarer_Pair_Direction`: Partnership of the declarer (NS/EW)
    - `Declarer_Direction`: Specific declarer position (N/E/S/W)
    - `BidSuit`: Contract strain (S/H/D/C/N)
    - `Probs_*_*_*_*`: Probability matrices for all contract combinations

    Output columns:
    - `Prob_Taking_0` through `Prob_Taking_13`: Trick-taking probabilities (pl.Float32)

    Returns:
    - DataFrame with added contract-specific trick probability columns
    """
    pd_pairs = [("NS", "N"), ("NS", "S"), ("EW", "E"), ("EW", "W")]
    suits = list("CDHSN")
    
    # Pre-compute which probability columns exist for better performance
    existing_prob_cols = {col for col in df.columns if col.startswith('Probs_')}

    def build_prob_expr(current_t: int) -> pl.Expr:
        terms: list[pl.Expr] = []
        for pair, decl in pd_pairs:
            for s in suits:
                prob_col = f"Probs_{pair}_{decl}_{s}_{current_t}"
                if prob_col in existing_prob_cols:  # Use pre-computed set
                    terms.append(
                        pl.when(
                            (pl.col("Declarer_Pair_Direction") == pair)
                            & (pl.col("Declarer_Direction") == decl)
                            & (pl.col("BidSuit") == s)
                        ).then(pl.col(prob_col)).otherwise(pl.lit(0.0, dtype=pl.Float32))
                    )
        if not terms:
            return pl.lit(None).alias(f"Prob_Taking_{current_t}")
        return (
            pl.when(pl.col("BidSuit").is_null()).then(None).otherwise(pl.sum_horizontal(terms)).alias(f"Prob_Taking_{current_t}")
        )

    return df.with_columns([build_prob_expr(t) for t in range(14)])


def add_declarer_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Create EV-based and computed declarer scores.

    Purpose:
    - Calculate declarer-specific scores using expected value methodology
    - Compute actual contract scores based on bid level, suit, tricks, and conditions
    - Generate declarer performance metrics for comparative analysis
    - Support declarer-focused scoring and evaluation

    Parameters:
    - df: DataFrame containing contract and performance information

    Input columns:
    - `EV_Score_Col_Declarer`: Column reference for declarer's expected value score
    - `BidLvl`: Contract bid level (1-7)
    - `BidSuit`: Contract suit (S/H/D/C/N)
    - `Tricks`: Actual tricks taken by declarer
    - `Vul_Declarer`: Declarer vulnerability flag
    - `Dbl`: Doubling indicator

    Output columns:
    - `EV_Score_Declarer`: Expected value score for declarer (pl.Float32)
    - `Computed_Score_Declarer`: Computed actual score for declarer (pl.Int32)
    - `Computed_Score_Declarer2`: Placeholder computed score variant (pl.Int32)

    Returns:
    - DataFrame with added declarer scoring columns
    """
    all_scores_d, _, _ = precompute_contract_score_tables()
    # NOTE: This block intentionally uses struct.map_elements for dynamic column selection.
    # Todo: replace with a pure-Polars approach (mask-based or index-based take) to remove Python row-wise code.
    return df.with_columns([
        pl.when(pl.col('EV_Score_Col_Declarer').is_null()).then(None).otherwise(
            pl.struct(['EV_Score_Col_Declarer'] + EV_SCORE_COLUMNS
            ).map_elements(lambda x: x[x['EV_Score_Col_Declarer']] if x['EV_Score_Col_Declarer'] is not None else None, return_dtype=pl.Float32)
        ).alias('EV_Score_Declarer'),
        # NOTE: This block intentionally uses a Python dict lookup for score computation.
        # Todo: replace with a Polars-native lookup/join on (BidLvl,BidSuit,Tricks,Vul_Declarer,Dbl) for performance.
        pl.struct(['BidLvl', 'BidSuit', 'Tricks', 'Vul_Declarer', 'Dbl']).map_elements(
            lambda x: all_scores_d.get(tuple(x.values()), None), return_dtype=pl.Int16
        ).alias('Computed_Score_Declarer'),
        pl.lit(None).alias('Computed_Score_Declarer2'),
    ])


def add_trick_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add flags for OverTricks/JustMade/UnderTricks based on `Result`.

    Purpose:
    - Create boolean indicators for contract performance outcomes
    - Classify contracts as making with overtricks, just making, or failing
    - Support statistical analysis of declarer play effectiveness
    - Enable filtering and grouping by contract outcome types

    Parameters:
    - df: DataFrame containing contract result information

    Input columns:
    - `Result`: Contract result (positive=overtricks, 0=just made, negative=undertricks)

    Output columns:
    - `OverTricks`: Boolean flag for contracts with overtricks (pl.Boolean)
    - `JustMade`: Boolean flag for contracts just made (pl.Boolean)
    - `UnderTricks`: Boolean flag for failed contracts (pl.Boolean)

    Returns:
    - DataFrame with added contract outcome flag columns
    """
    return df.with_columns([
        (pl.col('Result') > 0).alias('OverTricks'),
        (pl.col('Result') == 0).alias('JustMade'),
        (pl.col('Result') < 0).alias('UnderTricks'),
    ])


def add_player_ratings(df: pl.DataFrame) -> pl.DataFrame:
    """Add simple rating columns based on aggregated row features.

    Purpose:
    - Calculate player performance ratings based on actual vs predicted performance
    - Generate position-specific ratings for declarers and defenders
    - Create skill assessments using aggregated performance metrics
    - Support player ranking and performance tracking

    Parameters:
    - df: DataFrame containing player performance and position information

    Input columns:
    - `DD_Tricks_Diff`: Actual vs double-dummy trick difference
    - `Defender_Par_GE`: Defensive par performance indicator
    - `Declarer_ID`: Identifier for declaring player
    - `OnLead`: Identifier for opening leader
    - `NotOnLead`: Identifier for third hand defender

    Output columns:
    - `Declarer_Rating`: Average declarer performance rating (pl.Float32)
    - `Defender_OnLead_Rating`: Average opening leader performance rating (pl.Float32)
    - `Defender_NotOnLead_Rating`: Average third hand defender performance rating (pl.Float32)

    Returns:
    - DataFrame with added player rating columns
    """
    return df.with_columns([
        pl.col('DD_Tricks_Diff').cast(pl.Float32).mean().over('Declarer_ID').alias('Declarer_Rating'),
        pl.col('Defender_Par_GE').cast(pl.Float32).mean().over('OnLead').alias('Defender_OnLead_Rating'),
        pl.col('Defender_Par_GE').cast(pl.Float32).mean().over('NotOnLead').alias('Defender_NotOnLead_Rating'),
    ])


def compute_matchpoints_for_scores(df: pl.DataFrame, score_columns: list[str]) -> pl.DataFrame:
    """Compute matchpoints for a set of score columns in batches.

    Purpose:
    - Calculate matchpoint scores for multiple score columns efficiently
    - Compare each score column against actual achieved scores
    - Generate matchpoint rankings for theoretical and analytical score comparisons
    - Support batch processing for performance optimization

    Parameters:
    - df: input DataFrame containing score information and grouping keys
    - score_columns: column names to evaluate against Score_NS/EW

    Input columns:
    - Score columns specified in score_columns parameter
    - `Score_NS`, `Score_EW`: Actual partnership scores for comparison
    - Grouping columns for matchpoint calculation context

    Output columns:
    - `MP_{col}`: Matchpoint scores for each evaluated column (pl.Float32)

    Returns:
    - DataFrame with added matchpoint columns for all specified score columns
    """
    out = df
    
    # Note: DD_Score and Par columns are intentionally NOT included here because they
    # get more sophisticated computation in compute_dd_score_percentages() and 
    # compute_par_percentages() which use a proper lookup-based algorithm.
    # Including them here would create duplicate columns with _right suffix.
    all_columns = list(score_columns)  # Make a copy to avoid modifying the input
    
    # Filter to only include columns that exist in the DataFrame and don't already have MP columns
    existing_columns = []
    for col in all_columns:
        if col in out.columns and f'MP_{col}' not in out.columns:
            existing_columns.append(col)
    
    if not existing_columns:
        return out  # Nothing to process
    
    # Process in batches
    batch_size = 50
    for i in range(0, len(existing_columns), batch_size):
        batch = existing_columns[i:i+batch_size]
        
        batch_expressions = []
        for col in batch:
            comparison_col = 'Score_NS' if (col.endswith('_NS') or (len(col)>0 and col[-1] in 'NS')) else 'Score_EW'
            batch_expressions.append(
                pl.when(pl.col(col) > pl.col(comparison_col)).then(1.0)
                .when(pl.col(col) == pl.col(comparison_col)).then(0.5)
                .otherwise(pl.lit(0.0, dtype=pl.Float32)).sum().over(['session_id','PBN','Board']).alias(f'MP_{col}')
            )
        
        out = out.with_columns(batch_expressions)
    
    return out


def compute_matchpoints(df: pl.DataFrame, all_score_columns: list[str], batch_size: int = 50) -> pl.DataFrame:
    """Compute matchpoints for score columns with batching plus declarer and max sets.

    Purpose:
    - Calculate comprehensive matchpoint scores for all theoretical score combinations
    - Process large numbers of score columns efficiently using batching
    - Generate both partnership-based and declarer-specific matchpoint rankings
    - Support advanced scoring analysis and contract evaluation

    Parameters:
    - df: input DataFrame containing score columns and grouping keys `session_id`, `PBN`, `Board`
    - all_score_columns: core score columns to process (e.g., DD and EV by level/strain/dir)
    - batch_size: number of columns per batch for computation

    Input columns:
    - All score columns specified in all_score_columns parameter
    - `Score_NS`, `Score_EW`: Actual partnership scores
    - `Score_Declarer`: Actual declarer score
    - `session_id`, `PBN`, `Board`: Grouping keys for matchpoint context

    Output columns:
    - `MP_{col}`: Matchpoint scores for core columns (pl.Float32)
    - `MP_{col}_Declarer`: Declarer-specific matchpoint scores (pl.Float32)
    - `MP_{col}_Max_{NS|EW}`: Maximum matchpoint scores by partnership (pl.Float32)

    Returns:
    - DataFrame with comprehensive matchpoint scoring for all specified columns

    Behavior:
    - Processes core columns in batches against Score_NS/EW using suffix/seat heuristic
    - Then computes for declarer columns against Score_Declarer
    - Then computes for max columns against Score_NS/EW.

    Output columns:
    - For each input column c: `MP_{c}`; plus `MP_` for declarer and max sets.
    """
    out = df
    all_columns = all_score_columns + ['DD_Score_NS', 'DD_Score_EW', 'Par_NS', 'Par_EW']
    for i in range(0, len(all_columns), batch_size):
        batch = all_columns[i:i + batch_size]
        out = compute_matchpoints_for_scores(out, batch)

    declarer_columns = ['DD_Score_Declarer', 'Par_Declarer', 'EV_Score_Declarer', 'EV_Max_Declarer']
    out = compute_matchpoints_for_scores(out, declarer_columns)

    max_columns = ['EV_Max_NS', 'EV_Max_EW']
    out = compute_matchpoints_for_scores(out, max_columns)

    return out

def compute_dd_score_percentages(df: pl.DataFrame) -> pl.DataFrame:
    """Compute DD_Score_Pct_{NS|EW} using pure Polars operations (no map_elements).

    Purpose:
    - Calculate percentile rankings for double-dummy scores across all boards
    - Generate frequency-based percentages for theoretical score performance
    - Create board-specific score distribution analysis using unique session/section/board results
    - Support comparative analysis of double-dummy vs actual performance

    Parameters:
    - df: DataFrame containing actual and double-dummy score information

    Input columns:
    - `Score_NS`, `Score_EW`: Actual partnership scores achieved
    - `DD_Score_NS`, `DD_Score_EW`: Double-dummy theoretical scores
    - `Board`: Board identifier for grouping
    - `session_id`: Session identifier (required for unique result identification)
    - `section_id`: Section identifier (optional, improves uniqueness when available)
    - `MP_Top`: Maximum matchpoints for percentage calculation

    Output columns:
    - `MP_DD_Score_{NS|EW}`: Matchpoint scores for double-dummy results (pl.Float32)
    - `DD_Score_Pct_{NS|EW}`: Percentage rankings for double-dummy scores (pl.Float32)

    Returns:
    - DataFrame with added double-dummy score percentages and matchpoints
    
    Note:
    - Uses session_id + section_id + Board to identify unique results (each pair plays each board once per session)
    - Pure Polars implementation for maximum performance
    """
    import warnings
    warnings.filterwarnings("ignore")
    
    # Get unique columns for deduplication
    unique_cols = ['session_id', 'Board']
    if 'section_id' in df.columns:
        unique_cols.insert(1, 'section_id')
    
    # Process NS and EW separately
    for pair in ['NS', 'EW']:
        score_col = f'Score_{pair}'
        dd_score_col = f'DD_Score_{pair}'
        mp_col = f'MP_{dd_score_col}'
        pct_col = f'DD_Score_Pct_{pair}'
        
        # Get all actual scores per board for matchpoint calculation
        # Keep ALL scores including duplicates - each represents a result that can be beaten/tied
        # Group by session/section/board to keep different games separate
        score_unique_cols = unique_cols + [score_col]
        all_scores_per_board = (
            df.select(score_unique_cols)
            .unique()  # Only remove true duplicates (same session/section/board/score)
            .group_by(unique_cols)  # Group by session/section/board, not just board
            .agg([
                pl.col(score_col).alias('board_scores')
            ])
        )
        
        # Calculate matchpoints using a different approach - explode lists and aggregate
        # First, create a lookup table with matchpoints for each unique DD score per session/section/board
        dd_unique_scores = df.select(unique_cols + [dd_score_col]).unique()
        
        # Join with board scores and calculate matchpoints
        matchpoint_lookup = (
            dd_unique_scores
            .join(all_scores_per_board, on=unique_cols, how='left')
            .explode('board_scores')
            .group_by(unique_cols + [dd_score_col])
            .agg([
                # Count beats and ties
                (pl.col('board_scores') < pl.col(dd_score_col)).sum().cast(pl.Float32).alias('beats'),
                (pl.col('board_scores') == pl.col(dd_score_col)).sum().cast(pl.Float32).alias('ties'),
                pl.col('board_scores').count().alias('total_comparisons')
            ])
            .with_columns([
                # Calculate matchpoints and percentages
                (pl.col('beats') + pl.col('ties') * 0.5).alias(mp_col),
                ((pl.col('beats') + pl.col('ties') * 0.5) / 
                 pl.when(pl.col('total_comparisons') >= 1)
                 .then(pl.col('total_comparisons'))
                 .otherwise(pl.lit(1))).alias(pct_col)
            ])
            .select(unique_cols + [dd_score_col, mp_col, pct_col])
        )
        
        # Join the calculated values back to the main dataframe
        df = df.join(matchpoint_lookup, on=unique_cols + [dd_score_col], how='left')
        
        # Cast to Float32 for consistency
        df = df.with_columns([
            pl.col(mp_col).cast(pl.Float32),
            pl.col(pct_col).cast(pl.Float32)
        ])
    
    return df


def compute_par_percentages(df: pl.DataFrame) -> pl.DataFrame:
    """Compute Par-based percentages per side using pure Polars operations.

    Purpose:
    - Calculate percentile rankings for par score performance by board
    - Generate matchpoint scores for theoretical par results using unique session/section/board results
    - Create comparative metrics between actual and par performance
    - Support analysis of optimal vs achieved results

    Parameters:
    - df: DataFrame containing par score and actual score information

    Input columns:
    - `Par_NS`, `Par_EW`: Theoretical par scores for each partnership
    - `Score_NS`, `Score_EW`: Actual scores achieved by each partnership
    - `session_id`: Session identifier (required for unique result identification)
    - `section_id`: Section identifier (optional, improves uniqueness when available)
    - `MP_Top`: Maximum matchpoints for percentage calculation
    - `Board`: Board identifier for grouping

    Output columns:
    - `MP_Par_NS`, `MP_Par_EW`: Matchpoint scores for par results (pl.Float32)
    - `Par_Pct_NS`, `Par_Pct_EW`: Percentage rankings for par scores (pl.Float32)

    Returns:
    - DataFrame with added par-based percentage and matchpoint columns
    
    Note:
    - Uses session_id + section_id + Board to identify unique results (each pair plays each board once per session)
    - Pure Polars implementation for maximum performance
    """
    # Check if required Par columns exist
    required_cols = ['Par_NS', 'Par_EW', 'Score_NS', 'Score_EW', 'session_id', 'Board', 'MP_Top']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for par percentage calculation: {missing_cols}")
    
    # Define unique result identifier columns (same for both NS and EW)
    unique_cols = ['session_id', 'Board']
    if 'section_id' in df.columns:
        unique_cols.insert(1, 'section_id')
    
    # Build a list of all matchpoint lookup tables first
    all_lookups = []
    
    for pair in ['NS', 'EW']:
        score_col = f'Score_{pair}'
        par_col = f'Par_{pair}'
        mp_col = f'MP_{par_col}'
        
        # Get all actual scores per board for matchpoint calculation  
        # Keep ALL scores including duplicates - each represents a result that can be beaten/tied
        # Group by session/section/board to keep different games separate
        score_unique_cols = unique_cols + [score_col]
        all_scores_per_board = (
            df.select(score_unique_cols)
            .unique()  # Only remove true duplicates (same session/section/board/score)
            .group_by(unique_cols)  # Group by session/section/board, not just board
            .agg([
                pl.col(score_col).alias('board_scores')
            ])
        )
        
        # Calculate matchpoints using a similar approach to DD scores
        # First, create a lookup table with matchpoints for each unique Par score per session/section/board
        par_unique_scores = df.select(unique_cols + [par_col]).unique()
        
        # Join with board scores and calculate matchpoints
        matchpoint_lookup = (
            par_unique_scores
            .join(all_scores_per_board, on=unique_cols, how='left')
            .explode('board_scores')
            .group_by(unique_cols + [par_col])
            .agg([
                # Count beats and ties
                (pl.col('board_scores') < pl.col(par_col)).sum().cast(pl.Float32).alias('beats'),
                (pl.col('board_scores') == pl.col(par_col)).sum().cast(pl.Float32).alias('ties'),
                pl.col('board_scores').count().alias('total_comparisons')
            ])
            .with_columns([
                # Calculate matchpoints and percentage based on actual board scores
                (pl.col('beats') + pl.col('ties') * 0.5).alias(mp_col),
                ((pl.col('beats') + pl.col('ties') * 0.5) / 
                 pl.when(pl.col('total_comparisons') >= 1)
                 .then(pl.col('total_comparisons'))
                 .otherwise(pl.lit(1))).alias(f'Par_Pct_{pair}')
            ])
            .select(unique_cols + [par_col, mp_col, f'Par_Pct_{pair}'])
        )
        
        all_lookups.append((matchpoint_lookup, unique_cols + [par_col], mp_col, f'Par_Pct_{pair}'))
    
    # Join all lookups to the original DataFrame
    result = df
    for lookup, join_cols, mp_col, pct_col in all_lookups:
        result = result.join(lookup, on=join_cols, how='left')
    
    # Cast matchpoint columns to Float32 for consistency
    # Percentage columns are already calculated in the lookup tables
    cast_columns = []
    for _, _, mp_col, pct_col in all_lookups:
        cast_columns.extend([
            pl.col(mp_col).cast(pl.Float32),
            pl.col(pct_col).cast(pl.Float32)
        ])
    
    return result.with_columns(cast_columns)


def compute_mp_percentage_from_score(col: str) -> pl.Expr:
    """Create matchpoint percentage expression from MP score column.

    Purpose:
    - Convert raw matchpoint scores to percentage values for standardized analysis
    - Handle matchpoint normalization with proper division by (MP_Top + 1)
    - Support percentage-based matchpoint calculations across different field sizes
    - Generate standardized percentage metrics for comparative analysis

    Parameters:
    - col: Column name suffix to create percentage expression for

    Input columns:
    - `MP_{col}`: Raw matchpoint score for the specified column
    - `MP_Top`: Maximum possible matchpoints for the session

    Output columns:
    - Returns expression for percentage calculation (not added to DataFrame directly)

    Returns:
    - pl.Expr: Polars expression for calculating matchpoint percentage (bounded 0-1)
    """
    # Ensure Float32 to avoid implicit Float64 upcast
    # Cap percentage at 1.0 to handle cases where MP scores exceed MP_Top
    return pl.min_horizontal([
        (pl.col(f'MP_{col}').cast(pl.Float32))
        / (pl.col('MP_Top').cast(pl.Float32) + pl.lit(1.0, dtype=pl.Float32)),
        pl.lit(1.0, dtype=pl.Float32)
    ]).cast(pl.Float32)


def compute_declarer_percentages(df: pl.DataFrame) -> pl.DataFrame:
    """Compute declarer-based MP percentages using existing `MP_*` columns.

    Purpose:
    - Calculate matchpoint percentages from declarer perspective for performance analysis
    - Convert raw matchpoint scores to standardized percentage metrics
    - Support declarer-focused analysis of double-dummy, par, and expected value performance
    - Enable comparative analysis across different field sizes and sessions

    Parameters:
    - df: DataFrame containing matchpoint score columns for declarer analysis

    Input columns:
    - `MP_DD_Score_Declarer`: Raw matchpoints for double-dummy declarer score
    - `MP_Par_Declarer`: Raw matchpoints for par score from declarer perspective  
    - `MP_EV_Score_Declarer`: Raw matchpoints for expected value declarer score
    - `MP_EV_Max_Declarer`: Raw matchpoints for maximum expected value declarer score
    - `MP_Top`: Maximum possible matchpoints for percentage calculation

    Output columns:
    - `MP_DD_Pct_Declarer`: Double-dummy score percentage (pl.Float32)
    - `MP_Par_Pct_Declarer`: Par score percentage (pl.Float32)
    - `MP_EV_Pct_Declarer`: Expected value score percentage (pl.Float32)
    - `MP_EV_Max_Pct_Declarer`: Maximum expected value percentage (pl.Float32)

    Returns:
    - DataFrame with added declarer percentage columns
    """
    return df.with_columns([
        compute_mp_percentage_from_score('DD_Score_Declarer').alias('MP_DD_Pct_Declarer'),
        compute_mp_percentage_from_score('Par_Declarer').alias('MP_Par_Pct_Declarer'),
        compute_mp_percentage_from_score('EV_Score_Declarer').alias('MP_EV_Pct_Declarer'),
        compute_mp_percentage_from_score('EV_Max_Declarer').alias('MP_EV_Max_Pct_Declarer'),
    ])


def compute_max_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Compute maxima across MP-derived columns and their percentages.

    Purpose:
    - Calculate maximum achievable matchpoint scores across all contract levels
    - Find best possible double-dummy and expected value scores for each partnership
    - Generate percentage equivalents of maximum scores for comparative analysis
    - Support optimal contract identification and performance assessment

    Parameters:
    - df: DataFrame containing matchpoint scores for all contract combinations

    Input columns:
    - `MP_DD_Score_{level}{strain}_{NS|EW}`: Double-dummy matchpoint scores by contract
    - `MP_EV_{NS|EW}_{declarer}_{strain}_{level}_{V|NV}`: Expected value matchpoint scores
    - `MP_Top`: Maximum possible matchpoints for percentage calculations

    Output columns:
    - `MP_DD_Score_Max_{NS|EW}`: Maximum double-dummy matchpoint score (pl.Float32)
    - `MP_EV_Max_{NS|EW}`: Maximum expected value matchpoint score (pl.Float32)
    - `DD_Pct_Max_{NS|EW}`: Maximum double-dummy percentage (pl.Float32)
    - `EV_Pct_Max_{NS|EW}`: Maximum expected value percentage (pl.Float32)

    Returns:
    - DataFrame with added maximum score and percentage columns
    """
    out = df.with_columns([
        pl.max_horizontal(pl.col('^MP_DD_Score_[1-7][SHDCN]_NS$')).alias('MP_DD_Score_Max_NS'),
        pl.max_horizontal(pl.col('^MP_DD_Score_[1-7][SHDCN]_EW$')).alias('MP_DD_Score_Max_EW'),
        pl.max_horizontal(pl.col('^MP_EV_NS_[NS]_[SHDCN]_[1-7]_(V|NV)$')).alias('MP_EV_Max_NS'),
        pl.max_horizontal(pl.col('^MP_EV_EW_[EW]_[SHDCN]_[1-7]_(V|NV)$')).alias('MP_EV_Max_EW'),
    ])
    out = out.with_columns([
        # Fix DD_Pct_Max to ensure it's bounded between 0 and 1
        pl.when(pl.col('MP_Top') > 0)
        .then(
            pl.min_horizontal([
                (pl.coalesce([pl.col('MP_DD_Score_Max_NS').cast(pl.Float32), pl.lit(0.0, dtype=pl.Float32)]).cast(pl.Float32))
                / pl.col('MP_Top').cast(pl.Float32),
                pl.lit(1.0, dtype=pl.Float32)
            ])
        )
        .otherwise(pl.lit(0.0, dtype=pl.Float32)).alias('DD_Pct_Max_NS'),
        pl.when(pl.col('MP_Top') > 0)
        .then(
            pl.min_horizontal([
                (pl.coalesce([pl.col('MP_DD_Score_Max_EW').cast(pl.Float32), pl.lit(0.0, dtype=pl.Float32)]).cast(pl.Float32))
                / pl.col('MP_Top').cast(pl.Float32),
                pl.lit(1.0, dtype=pl.Float32)
            ])
        )
        .otherwise(pl.lit(0.0, dtype=pl.Float32)).alias('DD_Pct_Max_EW'),
        compute_mp_percentage_from_score('EV_Max_NS').alias('EV_Pct_Max_NS'),
        compute_mp_percentage_from_score('EV_Max_EW').alias('EV_Pct_Max_EW'),
    ])
    return out


def compute_score_differences(df: pl.DataFrame) -> pl.DataFrame:
    """Compute difference metrics comparing achieved vs max percentages (DD/EV).

    Purpose:
    - Calculate performance gaps between actual results and optimal theoretical results
    - Measure how much matchpoint percentage was lost compared to best possible outcomes
    - Support performance analysis and improvement opportunity identification
    - Generate metrics for evaluating bidding and play effectiveness

    Parameters:
    - df: DataFrame containing actual and maximum percentage scores

    Input columns:
    - `Pct_{NS|EW}`: Actual partnership percentages achieved
    - `EV_Pct_Max_{NS|EW}`: Maximum expected value percentages theoretically achievable
    - `DD_Pct_Max_{NS|EW}`: Maximum double-dummy percentages with perfect play

    Output columns:
    - `EV_Pct_Max_Diff_{NS|EW}`: Actual minus maximum EV percentage (pl.Float32)
    - `DD_Pct_Max_Diff_{NS|EW}`: Actual minus maximum DD percentage (pl.Float32)  
    - `DD_EV_Pct_Max_Diff_{NS|EW}`: DD maximum minus EV maximum percentage (pl.Float32)

    Returns:
    - DataFrame with added performance difference columns
    """
    zero = pl.lit(0.0, dtype=pl.Float32)
    return df.with_columns([
        (pl.coalesce([pl.col('Pct_NS').cast(pl.Float32), zero]) - pl.coalesce([pl.col('EV_Pct_Max_NS').cast(pl.Float32), zero])).alias('EV_Pct_Max_Diff_NS'),
        (pl.coalesce([pl.col('Pct_EW').cast(pl.Float32), zero]) - pl.coalesce([pl.col('EV_Pct_Max_EW').cast(pl.Float32), zero])).alias('EV_Pct_Max_Diff_EW'),
        (pl.coalesce([pl.col('Pct_NS').cast(pl.Float32), zero]) - pl.coalesce([pl.col('DD_Pct_Max_NS').cast(pl.Float32), zero])).alias('DD_Pct_Max_Diff_NS'),
        (pl.coalesce([pl.col('Pct_EW').cast(pl.Float32), zero]) - pl.coalesce([pl.col('DD_Pct_Max_EW').cast(pl.Float32), zero])).alias('DD_Pct_Max_Diff_EW'),
        (pl.coalesce([pl.col('DD_Pct_Max_NS').cast(pl.Float32), zero]) - pl.coalesce([pl.col('EV_Pct_Max_NS').cast(pl.Float32), zero])).alias('DD_EV_Pct_Max_Diff_NS'),
        (pl.coalesce([pl.col('DD_Pct_Max_EW').cast(pl.Float32), zero]) - pl.coalesce([pl.col('EV_Pct_Max_EW').cast(pl.Float32), zero])).alias('DD_EV_Pct_Max_Diff_EW'),
    ])
def add_event_elo_ratings(df: pl.DataFrame) -> pl.DataFrame:
    """Deprecated: use `compute_event_start_end_elo_columns` directly.

    Purpose:
    - Provide backward compatibility for legacy Elo rating function calls
    - Redirect to the current implementation of event-level Elo calculations
    - Maintain existing API while encouraging use of updated function name

    Parameters:
    - df: DataFrame containing Elo rating information

    Input columns:
    - Same as compute_event_start_end_elo_columns

    Output columns:
    - Same as compute_event_start_end_elo_columns

    Returns:
    - DataFrame with event-level Elo rating columns (via delegation)
    """
    return compute_event_start_end_elo_columns(df)


class DealAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _initialize_required_columns(self) -> None:
        # No specific input column requirements for initialization
        
        operations = []
        if 'group_id' not in self.df.columns:
            operations.append(pl.lit(0).alias('group_id'))
        if 'session_id' not in self.df.columns:
            operations.append(pl.lit(0).alias('session_id'))
        # section_name removed from hrs df because it conflicts with true section_name from brs. brs does not have global_id or session_id so no conflict.
        # if 'section_name' not in self.df.columns:
        #    operations.append(pl.lit('').alias('section_name'))
        
        if operations:
            self.df = self._time_operation("initialize required columns", lambda df: df.with_columns(operations), self.df)
        
        # Assert columns were created
        assert 'group_id' in self.df.columns, "Column 'group_id' was not created"
        assert 'session_id' in self.df.columns, "Column 'session_id' was not created"
        # assert 'section_name' in self.df.columns, "Column 'section_name' was not created"

    def _add_dealer_column(self) -> None:
        # Assert required columns exist
        assert 'Board' in self.df.columns, "Required column 'Board' not found in DataFrame"
        
        if 'Dealer' not in self.df.columns:
            self.df = self._time_operation("create Dealer", add_dealer_column, self.df)
        
        # Assert column was created
        assert 'Dealer' in self.df.columns, "Column 'Dealer' was not created"

    def _add_vulnerability_columns(self) -> None:
        # Assert required columns exist for iVul creation
        assert ('Vul' in self.df.columns or 'Board' in self.df.columns), "Required column 'Vul' or 'Board' not found in DataFrame"
        
        if 'iVul' not in self.df.columns:
            if 'Vul' in self.df.columns:
                self.df = self._time_operation("create iVul from Vul", encode_vulnerability, self.df)
            else:
                self.df = self._time_operation("create iVul from Board", derive_numeric_vulnerability_from_board, self.df)
        
        # Assert iVul column was created
        assert 'iVul' in self.df.columns, "Column 'iVul' was not created"
        
        if 'Vul' not in self.df.columns:
            self.df = self._time_operation("create Vul from iVul", decode_vulnerability, self.df)
        
        assert 'Vul' in self.df.columns, "Column 'Vul' was not created"
        
        if 'Vul_NS' not in self.df.columns:
            self.df = self._time_operation("create Vul_NS/EW", add_pair_vulnerability_flags, self.df)
        
        # Assert Vul_NS and Vul_EW columns were created
        assert 'Vul_NS' in self.df.columns, "Column 'Vul_NS' was not created"
        assert 'Vul_EW' in self.df.columns, "Column 'Vul_EW' was not created"

    def _parse_hands_from_pbn(self) -> None:
        # Assert required columns exist
        assert 'PBN' in self.df.columns, "Required column 'PBN' not found in DataFrame"
        
        self.df = self._time_operation("parse_pbn_to_hands", parse_pbn_to_hands, self.df)
        self.df = self._time_operation("extract_suits_by_seat", extract_suits_by_seat, self.df)
        self.df = self._time_operation("parse_pbn_to_hands_list", parse_pbn_to_hands_list, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            assert f'Hand_{direction}' in self.df.columns, f"Column 'Hand_{direction}' was not created"
            for suit in "SHDC":
                assert f'Suit_{direction}_{suit}' in self.df.columns, f"Column 'Suit_{direction}_{suit}' was not created"
        assert 'Hands' in self.df.columns, "Column 'Hands' was not created"

    def perform_deal_augmentations(self) -> pl.DataFrame:
        """Main method to perform all deal augmentations"""
        t_start = time.time()
        logger.info(f"Starting deal augmentations")
        
        self._initialize_required_columns()
        self._add_dealer_column()
        self._add_vulnerability_columns()
        self._parse_hands_from_pbn()
        
        logger.info(f"Deal augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class HandAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.suit_quality_criteria = {
            "Biddable": lambda sl, hcp: sl.ge(5) | (sl.eq(4) & hcp.ge(3)),
            "Rebiddable": lambda sl, hcp: sl.ge(6) | (sl.eq(5) & hcp.ge(3)),
            "Twice_Rebiddable": lambda sl, hcp: sl.ge(7) | (sl.eq(6) & hcp.ge(3)),
            "Strong_Rebiddable": lambda sl, hcp: sl.ge(6) & hcp.ge(9),
            "Solid": lambda sl, hcp: hcp.ge(9),  # todo: 6 card requires ten
        }
        self.stopper_criteria = {
            "At_Best_Partial_Stop_In": lambda sl, hcp: (sl + hcp).lt(4),
            "Partial_Stop_In": lambda sl, hcp: (sl + hcp).ge(4),
            "Likely_Stop_In": lambda sl, hcp: (sl + hcp).ge(5),
            "Stop_In": lambda sl, hcp: hcp.ge(4) | (sl + hcp).ge(6),
            "At_Best_Stop_In": lambda sl, hcp: (sl + hcp).ge(7),
            "Two_Stops_In": lambda sl, hcp: (sl + hcp).ge(8),
        }

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _add_card_presence_indicators(self) -> None:
        # Assert required columns exist
        assert 'PBN' in self.df.columns, "Required column 'PBN' not found in DataFrame"
        
        if 'C_NSA' not in self.df.columns:
            self.df = self._time_operation("create C_NSA", add_card_presence_columns, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                for rank in 'AKQJT98765432':
                    assert f'C_{direction}{suit}{rank}' in self.df.columns, f"Column 'C_{direction}{suit}{rank}' was not created"

    def _compute_high_card_points(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                for rank in 'AKQJT98765432':
                    assert f'C_{direction}{suit}{rank}' in self.df.columns, f"Required column 'C_{direction}{suit}{rank}' not found in DataFrame"
        
        if 'HCP_N_C' not in self.df.columns:
            self.df = self._time_operation("create HCP", add_hcp_from_cards, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'HCP_{direction}_{suit}' in self.df.columns, f"Column 'HCP_{direction}_{suit}' was not created"

    def _compute_quick_tricks(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'Suit_{direction}_{suit}' in self.df.columns, f"Required column 'Suit_{direction}_{suit}' not found in DataFrame"
        
        if 'QT_N_C' not in self.df.columns:
            self.df = self._time_operation("create QT", add_quick_tricks, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'QT_{direction}_{suit}' in self.df.columns, f"Column 'QT_{direction}_{suit}' was not created"

    def _compute_quick_losers(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'Suit_{direction}_{suit}' in self.df.columns, f"Required column 'Suit_{direction}_{suit}' not found in DataFrame"
        
        if 'QL_N_S' not in self.df.columns:
            self.df = self._time_operation("create QL", add_quick_losers, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'QL_{direction}_{suit}' in self.df.columns, f"Column 'QL_{direction}_{suit}' was not created"

    def _compute_losing_trick_count(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'Suit_{direction}_{suit}' in self.df.columns, f"Required column 'Suit_{direction}_{suit}' not found in DataFrame"
        
        if 'LTC_N_S' not in self.df.columns:
            self.df = self._time_operation("create LTC", add_losing_trick_count, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'LTC_{direction}_{suit}' in self.df.columns, f"Column 'LTC_{direction}_{suit}' was not created"

    def _compute_suit_lengths(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'Suit_{direction}_{suit}' in self.df.columns, f"Required column 'Suit_{direction}_{suit}' not found in DataFrame"
        
        if 'SL_N_C' not in self.df.columns:
            self.df = self._time_operation("create SL_[NESW]_[SHDC]", add_suit_lengths, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'SL_{direction}_{suit}' in self.df.columns, f"Column 'SL_{direction}_{suit}' was not created"

    def _compute_partnership_suit_lengths(self) -> None:
        # Assert required columns exist
        for pair in ['NS', 'EW']:
            for suit in "SHDC":
                assert f'SL_{pair[0]}_{suit}' in self.df.columns, f"Required column 'SL_{pair[0]}_{suit}' not found in DataFrame"
                assert f'SL_{pair[1]}_{suit}' in self.df.columns, f"Required column 'SL_{pair[1]}_{suit}' not found in DataFrame"
        
        if 'SL_NS_C' not in self.df.columns:
            self.df = self._time_operation("create SL_(NS|EW)_[SHDC]", add_pair_suit_lengths, self.df)
        
        # Assert columns were created
        for pair in ['NS', 'EW']:
            for suit in "SHDC":
                assert f'SL_{pair}_{suit}' in self.df.columns, f"Column 'SL_{pair}_{suit}' was not created"

    def _build_suit_length_distributions(self) -> None:
        # Assert required columns exist
        for direction in 'NESW':
            for suit in 'CDHS':
                assert f'SL_{direction}_{suit}' in self.df.columns, f"Required column 'SL_{direction}_{suit}' not found in DataFrame"
        
        if 'SL_N_CDHS' not in self.df.columns:
            for d in 'NESW':
                self.df = self._time_operation(f"create SL_{d} arrays", add_suit_length_arrays, self.df, d)
        
        # Assert columns were created
        for direction in 'NESW':
            assert f'SL_{direction}_CDHS' in self.df.columns, f"Column 'SL_{direction}_CDHS' was not created"
            assert f'SL_{direction}_CDHS_SJ' in self.df.columns, f"Column 'SL_{direction}_CDHS_SJ' was not created"
            assert f'SL_{direction}_ML' in self.df.columns, f"Column 'SL_{direction}_ML' was not created"
            assert f'SL_{direction}_ML_SJ' in self.df.columns, f"Column 'SL_{direction}_ML_SJ' was not created"
            assert f'SL_{direction}_ML_I' in self.df.columns, f"Column 'SL_{direction}_ML_I' was not created"
            assert f'SL_{direction}_ML_I_SJ' in self.df.columns, f"Column 'SL_{direction}_ML_I_SJ' was not created"

    def _compute_distribution_points(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'SL_{direction}_{suit}' in self.df.columns, f"Required column 'SL_{direction}_{suit}' not found in DataFrame"
        
        if 'DP_N_C' not in self.df.columns:
            # Calculate individual suit DPs
            self.df = self._time_operation("create DP columns (distribution points)", add_distribution_points, self.df)
            self.df = self._time_operation("create DP columns (direction summaries)", add_direction_summaries, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'DP_{direction}_{suit}' in self.df.columns, f"Column 'DP_{direction}_{suit}' was not created"
            assert f'DP_{direction}' in self.df.columns, f"Column 'DP_{direction}' was not created"
        assert 'DP_NS' in self.df.columns, "Column 'DP_NS' was not created"
        assert 'DP_EW' in self.df.columns, "Column 'DP_EW' was not created"
        for suit in 'SHDC':
            assert f'DP_NS_{suit}' in self.df.columns, f"Column 'DP_NS_{suit}' was not created"
            assert f'DP_EW_{suit}' in self.df.columns, f"Column 'DP_EW_{suit}' was not created"

    def _compute_total_points(self) -> None:
        # Assert required columns exist
        for direction in 'NESW':
            for suit in 'SHDC':
                assert f'HCP_{direction}_{suit}' in self.df.columns, f"Required column 'HCP_{direction}_{suit}' not found in DataFrame"
                assert f'DP_{direction}_{suit}' in self.df.columns, f"Required column 'DP_{direction}_{suit}' not found in DataFrame"
        
        if 'Total_Points_N_C' not in self.df.columns:
            logger.warning("Todo: Don't forget to adjust Total_Points for singleton king and doubleton queen.")
            self.df = self._time_operation("create Total_Points", add_total_points, self.df)
        
        # Assert columns were created
        for direction in 'NESW':
            for suit in 'SHDC':
                assert f'Total_Points_{direction}_{suit}' in self.df.columns, f"Column 'Total_Points_{direction}_{suit}' was not created"
            assert f'Total_Points_{direction}' in self.df.columns, f"Column 'Total_Points_{direction}' was not created"
        assert 'Total_Points_NS' in self.df.columns, "Column 'Total_Points_NS' was not created"
        assert 'Total_Points_EW' in self.df.columns, "Column 'Total_Points_EW' was not created"

    def _find_longest_suits(self) -> None:
        # Assert required columns exist
        for direction in ['NS', 'EW']:
            for suit in ['S', 'H', 'D', 'C']:
                assert f'SL_{direction[0]}_{suit}' in self.df.columns, f"Required column 'SL_{direction[0]}_{suit}' not found in DataFrame"
                assert f'SL_{direction[1]}_{suit}' in self.df.columns, f"Required column 'SL_{direction[1]}_{suit}' not found in DataFrame"
        
        if 'SL_Max_NS' not in self.df.columns:
            self.df = self._time_operation("create SL_Max columns", add_max_suit_lengths, self.df)
        
        # Assert columns were created
        for direction in ['NS', 'EW']:
            assert f'SL_Max_{direction}' in self.df.columns, f"Column 'SL_Max_{direction}' was not created"

    def _evaluate_suit_quality(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'SL_{direction}_{suit}' in self.df.columns, f"Required column 'SL_{direction}_{suit}' not found in DataFrame"
                assert f'HCP_{direction}_{suit}' in self.df.columns, f"Required column 'HCP_{direction}_{suit}' not found in DataFrame"
        
        self.df = self._time_operation("create quality indicators", add_suit_quality_indicators, self.df, self.suit_quality_criteria, self.stopper_criteria)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                for series_type in {**self.suit_quality_criteria, **self.stopper_criteria}.keys():
                    assert f'{series_type}_{direction}_{suit}' in self.df.columns, f"Column '{series_type}_{direction}_{suit}' was not created"
        assert 'Forcing_One_Round' in self.df.columns, "Column 'Forcing_One_Round' was not created"
        assert 'Opponents_Cannot_Play_Undoubled_Below_2N' in self.df.columns, "Column 'Opponents_Cannot_Play_Undoubled_Below_2N' was not created"
        assert 'Forcing_To_2N' in self.df.columns, "Column 'Forcing_To_2N' was not created"
        assert 'Forcing_To_3N' in self.df.columns, "Column 'Forcing_To_3N' was not created"

    def _assess_hand_balance(self) -> None:
        # Assert required columns exist
        for direction in 'NESW':
            assert f'SL_{direction}_ML_SJ' in self.df.columns, f"Required column 'SL_{direction}_ML_SJ' not found in DataFrame"
            assert f'SL_{direction}_C' in self.df.columns, f"Required column 'SL_{direction}_C' not found in DataFrame"
            assert f'SL_{direction}_D' in self.df.columns, f"Required column 'SL_{direction}_D' not found in DataFrame"
        
        self.df = self._time_operation("create balanced indicators", add_balanced_indicators, self.df)
        
        # Assert columns were created
        for direction in 'NESW':
            assert f'Balanced_{direction}' in self.df.columns, f"Column 'Balanced_{direction}' was not created"

    def perform_hand_augmentations(self) -> pl.DataFrame:
        """Main method to perform all hand augmentations"""
        t_start = time.time()
        logger.info(f"Starting hand augmentations")
        
        self._add_card_presence_indicators()
        self._compute_high_card_points()
        self._compute_quick_tricks()
        self._compute_quick_losers()
        self._compute_losing_trick_count()
        self._compute_suit_lengths()
        self._compute_partnership_suit_lengths()
        self._build_suit_length_distributions()
        self._compute_distribution_points()
        self._compute_total_points()
        self._find_longest_suits()
        self._evaluate_suit_quality()
        self._assess_hand_balance()
        
        logger.info(f"Hand augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class DD_SD_Augmenter:
    def __init__(self, df: pl.DataFrame, hrs_cache_df: pl.DataFrame, sd_productions: int = 40, max_dd_adds: Optional[int] = None, max_sd_adds: Optional[int] = None, output_progress: Optional[bool] = True, progress: Optional[Any] = None, lock_func: Optional[Callable[..., pl.DataFrame]] = None):
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_dd_adds = max_dd_adds
        self.max_sd_adds = max_sd_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _process_scores_and_tricks(self) -> pl.DataFrame:
        # Assert required columns exist
        assert 'PBN' in self.df.columns, "Required column 'PBN' not found in DataFrame"
        assert 'Dealer' in self.df.columns, "Required column 'Dealer' not found in DataFrame"
        assert 'Vul' in self.df.columns, "Required column 'Vul' not found in DataFrame"
        all_scores_d, scores_d, scores_df = self._time_operation("calculate_scores", precompute_contract_score_tables)
        
        # Calculate double dummy scores first
        dd_df, unique_dd_tables_d = self._time_operation(
            "compute_dd_scores", 
            compute_dd_trick_tables, 
            self.df, self.hrs_cache_df, self.max_dd_adds, self.output_progress, self.progress
        )

        if not dd_df.is_empty():
            self.hrs_cache_df = self._time_operation("update cache with DD", update_hand_records_cache, self.hrs_cache_df, dd_df)
        
        # Calculate par scores using the double dummy results
        par_df = self._time_operation(
            "compute_par_scores",
            compute_par_scores_for_missing,
            self.df, self.hrs_cache_df, unique_dd_tables_d
        )

        if not par_df.is_empty():
            self.hrs_cache_df = self._time_operation("update cache with Par", update_hand_records_cache, self.hrs_cache_df, par_df)

        sd_dfs_d, sd_df = self._time_operation(
            "calculate_sd_probs",
            estimate_sd_trick_distributions_for_df,
            self.df, self.hrs_cache_df, self.sd_productions, self.max_sd_adds, self.progress
        )

        if not sd_df.is_empty():
            self.hrs_cache_df = self._time_operation("update cache with SD", update_hand_records_cache, self.hrs_cache_df, sd_df)

        self.df = self.df.join(self.hrs_cache_df, on=['PBN','Dealer','Vul'], how='inner') # on='PBN', how='left' or on=['PBN','Dealer','Vul'], how='inner'

        # create DD_(NS|EW)_[SHDCN] which is the max of NS or EW for each strain
        self.df = self.df.with_columns(
            pl.max_horizontal(f"DD_{pair[0]}_{strain}",f"DD_{pair[1]}_{strain}").alias(f"DD_{pair}_{strain}")
            for pair in ['NS','EW']
            for strain in "SHDCN"
        )
        
        # Create individual direction DD score columns if they don't exist
        if f'DD_Score_1C_N' not in self.df.columns:
            self.df = self._time_operation("create individual DD scores", add_dd_scores_basic, self.df, scores_d)
        
        # create DD_Score_(level)(strain)_(NS|EW) which is the max of NS or EW for each contract
        self.df = self.df.with_columns([
            pl.max_horizontal(f"DD_Score_{level}{strain}_{pair[0]}", f"DD_Score_{level}{strain}_{pair[1]}").alias(f"DD_Score_{level}{strain}_{pair}")
            for level in range(1, 8)
            for strain in 'CDHSN'
            for pair in ['NS', 'EW']
        ])

        self.df = self._time_operation(
            "calculate_sd_expected_values",
            add_single_dummy_expected_values,
            self.df, scores_df
        )

        best_contracts_df = self._time_operation("identify best contracts by EV", identify_best_contracts_by_ev, self.df)
        assert self.df.height == best_contracts_df.height, f"{self.df.height} != {best_contracts_df.height}"
        self.df = pl.concat([self.df, best_contracts_df], how='horizontal')
        del best_contracts_df        

        # Assert columns were created
        assert 'DD_N_S' in self.df.columns, "Column 'DD_N_S' was not created"
        assert 'DD_NS_S' in self.df.columns, "Column 'DD_NS_S' was not created"
        assert 'DD_Score_1C_N' in self.df.columns, "Column 'DD_Score_1C_N' was not created"
        assert 'DD_Score_1C_NS' in self.df.columns, "Column 'DD_Score_1C_NS' was not created"

        return self.df, self.hrs_cache_df #, scores_df

    def perform_dd_sd_augmentations(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        if self.lock_func is None:
            self.df, self.hrs_cache_df = self.perform_dd_sd_augmentations_queue_up()
        else:
            self.df, self.hrs_cache_df = self.lock_func(self, self.perform_dd_sd_augmentations_queue_up)
        return self.df, self.hrs_cache_df

    def perform_dd_sd_augmentations_queue_up(self) -> pl.DataFrame:
        """Main method to perform all double dummy and single dummy augmentations"""
        t_start = time.time()
        logger.info("Starting DD/SD trick augmentations")
        
        self.df, self.hrs_cache_df = self._process_scores_and_tricks()
        
        logger.info(f"DD/SD trick augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df, self.hrs_cache_df


class AllContractsAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_ct_types(self) -> None:
            self.df = self._time_operation(
                "create CT columns",
            add_contract_types,
            self.df,
            )

    # create CT boolean columns from CT columns
    def _create_ct_booleans(self) -> None:
            self.df = self._time_operation(
                "create CT boolean columns",
            add_contract_type_flags,
            self.df,
            )

    def perform_all_contracts_augmentations(self) -> pl.DataFrame:
        """Main method to perform AllContracts augmentations"""
        t_start = time.time()
        logger.info("Starting AllContracts augmentations")

        # uses class-local implementation; consider swapping to global `create_ct_types(self.df)` later
        self._create_ct_types()
        self._create_ct_booleans()

        logger.info(f"AllContracts augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


def add_score_calculations(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure Score/Score_NS/Score_EW exist using contract parts or existing scores.
    
    Purpose:
    - Create consistent scoring columns from available contract information
    - Bridge different data sources that may have different score column formats
    - Calculate scores from scratch if only contract components are available
    
    Parameters:
    - df: DataFrame containing contract and/or scoring information
    
    Returns:
    - DataFrame with added/normalized score columns
    
    Input columns (one of):
    - `Score_NS`, `Score_EW`, `Declarer_Pair_Direction`: Pre-calculated scores
    - `BidLvl`, `BidSuit`, `Tricks`, `Vul_Declarer`, `Dbl`: Contract components
    
    Output columns:
    - `Score`: Declarer-perspective score
    - `Score_NS`: North-South score
    - `Score_EW`: East-West score
    """
    out = df
    if 'Score' not in out.columns:
        if 'Score_NS' in out.columns:
            assert 'Score_EW' in out.columns, "Score_EW does not exist but Score_NS does."
            out = out.with_columns(
                pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
                .then(pl.col('Score_NS'))
                .otherwise(pl.col('Score_EW'))
                .alias('Score')
            )
        else:
            all_scores_d, _, _ = precompute_contract_score_tables()
            out = out.with_columns(
                pl.struct(['BidLvl', 'BidSuit', 'Tricks', 'Vul_Declarer', 'Dbl'])
                .map_elements(lambda x: all_scores_d.get(tuple(x.values()), None), return_dtype=pl.Int16)
                .alias('Score')
            )

    if 'Score_NS' not in out.columns:
        out = out.with_columns([
            pl.when(pl.col('Declarer_Pair_Direction').eq('NS')).then(pl.col('Score')).otherwise(-pl.col('Score')).alias('Score_NS'),
            pl.when(pl.col('Declarer_Pair_Direction').eq('EW')).then(pl.col('Score')).otherwise(-pl.col('Score')).alias('Score_EW'),
        ])
    return out


def create_score_diff_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create Par/EV and trick difference columns using vectorized expressions.

    Purpose:
    - Calculate performance gaps between actual and theoretical optimal results
    - Generate difference metrics for par scores, expected values, and trick counts
    - Create standardized performance evaluation columns
    - Support analysis of bidding and play effectiveness

    Parameters:
    - df: DataFrame containing actual and theoretical performance data

    Input columns:
    - `Score_NS`, `Score_EW`: Actual partnership scores achieved
    - `Par_NS`, `Par_EW`: Theoretical par scores
    - `EV_Max_NS`, `EV_Max_EW`: Maximum expected value scores
    - `Tricks`: Actual tricks taken by declarer
    - `DD_Tricks`: Double-dummy predicted tricks

    Output columns:
    - `Par_Diff_NS`, `Par_Diff_EW`: Score difference vs par (pl.Int16)
    - `DD_Tricks_Diff`: Trick difference vs double-dummy (pl.Int8)
    - `EV_Max_Diff_NS`, `EV_Max_Diff_EW`: Score difference vs maximum EV (pl.Float32)

    Returns:
    - DataFrame with added performance difference columns
    """
    out = df.with_columns([
        (pl.col('Score_NS') - pl.col('Par_NS')).cast(pl.Int16).alias('Par_Diff_NS'),
        (pl.col('Score_EW') - pl.col('Par_EW')).cast(pl.Int16).alias('Par_Diff_EW'),
        (pl.col('Tricks').cast(pl.Int8) - pl.col('DD_Tricks').cast(pl.Int8)).cast(pl.Int8).alias('DD_Tricks_Diff'),
        (pl.col('Score_NS') - pl.col('EV_Max_NS')).cast(pl.Float32).alias('EV_Max_Diff_NS'),
        (pl.col('Score_EW') - pl.col('EV_Max_EW')).cast(pl.Float32).alias('EV_Max_Diff_EW'),
    ])
    # Ensure EW diff is the negative of NS if desired
    out = out.with_columns((-pl.col('Par_Diff_NS')).cast(pl.Int16).alias('Par_Diff_EW'))
    return out


def add_trick_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add OverTricks/JustMade/UnderTricks boolean columns from Result.

    Purpose:
    - Create boolean indicators for different contract outcome categories
    - Enable filtering and analysis by contract performance type
    - Support statistical classification of declarer play results
    - Provide categorical flags for contract success analysis

    Parameters:
    - df: DataFrame containing contract result information

    Input columns:
    - `Result`: Contract result (positive=overtricks, 0=just made, negative=undertricks)

    Output columns:
    - `OverTricks`: Boolean flag for contracts with overtricks (pl.Boolean)
    - `JustMade`: Boolean flag for contracts just made (pl.Boolean)
    - `UnderTricks`: Boolean flag for failed contracts (pl.Boolean)

    Returns:
    - DataFrame with added contract outcome boolean columns
    """
    return df.with_columns([
        (pl.col('Result') > 0).alias('OverTricks'),
        (pl.col('Result') == 0).alias('JustMade'),
        (pl.col('Result') < 0).alias('UnderTricks'),
    ])


def compute_group_matchpoints(series_list: list[pl.Series]) -> pl.Series:
    """Compute matchpoints for a group by comparing each value against all others.

    Purpose:
    - Calculate matchpoint totals by comparing each value to all other scores in group
    - Treat ties as half a point, wins as one point, losses as zero
    - Support per-board/per-session matchpoint computations

    Parameters:
    - series_list: List containing [col_values, score_ns_values]

    Input columns:
    - None (operates on provided Series values directly)

    Output columns:
    - Returns a Series of matchpoint totals (not added to DataFrame directly)

    Returns:
    - pl.Series: Matchpoint scores for each value vs the reference vector
    """
    col_values = series_list[0]
    score_ns_values = series_list[1]
    if col_values.is_null().sum() > 0:
        logger.warning(f"Null values in col_values: {col_values.is_null().sum()}")
    
    # Fill nulls to handle sitout/adjusted scores
    score_ns_values = score_ns_values.fill_null(0.0)
    col_values = col_values.fill_null(0.0)
    return pl.Series([
        sum(1.0 if val > score else 0.5 if val == score else 0.0 
            for score in score_ns_values)
        for val in col_values
    ])
def compute_all_score_matchpoints(df: pl.DataFrame, all_score_columns: list[str]) -> pl.DataFrame:
    """Calculate matchpoints for all score columns in batches to prevent memory issues.

    Purpose:
    - Calculate comprehensive matchpoint scores for large numbers of score columns
    - Process columns in batches to manage memory usage efficiently
    - Generate matchpoint rankings for all theoretical and actual score comparisons
    - Support large-scale scoring analysis and performance evaluation

    Parameters:
    - df: Input DataFrame containing score information and grouping keys
    - all_score_columns: List of score columns to process for matchpoint calculation

    Input columns:
    - All score columns specified in all_score_columns parameter
    - `Score_NS`, `Score_EW`: Actual partnership scores for comparison
    - Grouping columns required for matchpoint context

    Output columns:
    - `MP_{col}`: Matchpoint scores for each processed column (pl.Float32)

    Returns:
    - DataFrame with matchpoint columns added for all specified score columns
    """
    logger.info(f"Processing {len(all_score_columns)} score columns in batches...")
    
    # Process the main score columns (compute_matchpoints_for_scores will add DD_Score_NS/EW and Par_NS/EW internally)
    out = df
    if all_score_columns:  # Only process if there are columns to process
        batch_size = 50  # Process 50 columns at a time
        for i in range(0, len(all_score_columns), batch_size):
            batch = all_score_columns[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_score_columns) + batch_size - 1)//batch_size}: {len(batch)} columns")
            out = compute_matchpoints_for_scores(out, batch)
    else:
        # If no custom score columns, just process the standard ones
        out = compute_matchpoints_for_scores(out, [])
    
    # Calculate matchpoints for declarer columns
    declarer_columns = ['DD_Score_Declarer', 'Par_Declarer', 'EV_Score_Declarer', 'EV_Max_Declarer']
    # Filter to only include columns that exist in the DataFrame
    existing_declarer_columns = [col for col in declarer_columns if col in out.columns]
    if existing_declarer_columns:
        out = compute_matchpoints_for_scores(out, existing_declarer_columns)
    
    # Calculate matchpoints for max columns
    max_columns = ['EV_Max_NS', 'EV_Max_EW']
    # Filter to only include columns that exist in the DataFrame
    existing_max_columns = [col for col in max_columns if col in out.columns]
    if existing_max_columns:
        out = compute_matchpoints_for_scores(out, existing_max_columns)
    
    return out


def create_result_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create Result and Tricks columns from Contract or existing data.

    Purpose:
    - Generate contract result and trick count columns from available contract data
    - Calculate performance metrics from bid level and actual trick information
    - Create standardized result columns for bridge analysis
    - Handle derivation from existing contract or trick data where available

    Parameters:
    - df: DataFrame containing contract information

    Input columns:
    - `Contract`: Contract string for analysis (required)
    - `Tricks`: Actual tricks taken (optional, derived if missing)

    Output columns:
    - `Result`: Contract result (tricks over/under bid) (pl.Int8)
    - `Tricks`: Actual tricks taken by declarer (pl.UInt8)

    Returns:
    - DataFrame with Result and Tricks columns added
    """
    # Assert required columns exist
    assert 'Contract' in df.columns, "Required column 'Contract' not found in DataFrame"
    
    out = df
    if 'Result' not in out.columns:
        if 'Tricks' in out.columns:
            out = derive_result_from_tricks(out)
        else:
            out = derive_result_from_contract(out)

    if 'Tricks' not in out.columns:
        out = derive_tricks_from_contract(out)
    
    # Assert columns were created
    assert 'Result' in out.columns, "Column 'Result' was not created"
    assert 'Tricks' in out.columns, "Column 'Tricks' was not created"
    
    return out


class FinalContractAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        # note that polars exprs are allowed in dict values.
        self.get_vulnerability_conditions = {
            'NS': pl.col('Vul').is_in(['N_S', 'Both']),
            'EW': pl.col('Vul').is_in(['E_W', 'Both'])
        }

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _standardize_contract_format(self) -> None:
        # Assert required columns exist
        assert 'Contract' in self.df.columns, "Required column 'Contract' not found in DataFrame"
        
        self.df = self._time_operation(
            "normalize_contract_columns",
            normalize_contract_columns,
            self.df,
        )
        
        # Assert columns were created
        assert 'BidLvl' in self.df.columns, "Column 'BidLvl' was not created"
        assert 'BidSuit' in self.df.columns, "Column 'BidSuit' was not created"
        assert 'Dbl' in self.df.columns, "Column 'Dbl' was not created"
        assert 'Declarer_Direction' in self.df.columns, "Column 'Declarer_Direction' was not created"
        assert 'Declarer_Pair_Direction' in self.df.columns, "Column 'Declarer_Pair_Direction' was not created"
        assert 'Vul_Declarer' in self.df.columns, "Column 'Vul_Declarer' was not created"

    # create ContractType column using final contract
    def _classify_contract_types(self) -> None:
        # Assert required columns exist
        assert 'Contract' in self.df.columns, "Required column 'Contract' not found in DataFrame"
        assert 'BidLvl' in self.df.columns, "Required column 'BidLvl' not found in DataFrame"
        assert 'BidSuit' in self.df.columns, "Required column 'BidSuit' not found in DataFrame"
        
        self.df = self._time_operation(
            "create_contract_types",
            add_contract_type,
            self.df
        )
        
        # Assert column was created
        assert 'ContractType' in self.df.columns, "Column 'ContractType' was not created"

    def _build_player_ids(self) -> None:
        
        if 'Player_ID_N' not in self.df.columns:
            assert 'Pair_IDs_NS' in self.df.columns, "Required column 'Pair_IDs_NS' not found in DataFrame"
            assert 'Pair_IDs_EW' in self.df.columns, "Required column 'Pair_IDs_EW' not found in DataFrame"
            self.df = self._time_operation(
                "add_player_ids",
                add_player_ids,
                self.df
            )
        
        # Assert column was created
        assert 'Player_ID_N' in self.df.columns, "Column 'Player_ID_N' was not created"
        assert 'Player_ID_E' in self.df.columns, "Column 'Player_ID_E' was not created"
        assert 'Player_ID_S' in self.df.columns, "Column 'Player_ID_S' was not created"
        assert 'Player_ID_W' in self.df.columns, "Column 'Player_ID_W' was not created"

    def _build_player_names(self) -> None:
        
        if 'Player_Name_N' not in self.df.columns:
            assert 'Pair_Names_NS' in self.df.columns, "Required column 'Pair_Names_NS' not found in DataFrame"
            assert 'Pair_Names_EW' in self.df.columns, "Required column 'Pair_Names_EW' not found in DataFrame"
            self.df = self._time_operation(
                "add_player_names",
                add_player_names,
                self.df
            )
        
        # Assert column was created
        assert 'Player_Name_N' in self.df.columns, "Column 'Player_Name_N' was not created"
        assert 'Player_Name_E' in self.df.columns, "Column 'Player_Name_E' was not created"
        assert 'Player_Name_S' in self.df.columns, "Column 'Player_Name_S' was not created"
        assert 'Player_Name_W' in self.df.columns, "Column 'Player_Name_W' was not created"

    def _build_partnership_ids(self) -> None:
        
        if 'Pair_IDs_NS' not in self.df.columns:
            assert 'Player_ID_N' in self.df.columns, "Required column 'Player_ID_N' not found in DataFrame"
            assert 'Player_ID_E' in self.df.columns, "Required column 'Player_ID_E' not found in DataFrame"
            assert 'Player_ID_S' in self.df.columns, "Required column 'Player_ID_S' not found in DataFrame"
            assert 'Player_ID_W' in self.df.columns, "Required column 'Player_ID_W' not found in DataFrame"
            self.df = self._time_operation(
                "add_pair_ids",
                add_pair_ids,
                self.df
            )
        
        # Assert column was created
        assert 'Pair_IDs_NS' in self.df.columns, "Column 'Pair_IDs_NS' was not created"
        assert 'Pair_IDs_EW' in self.df.columns, "Column 'Pair_IDs_EW' was not created"

    def _build_partnership_names(self) -> None:
        
        if 'Pair_Names_NS' not in self.df.columns:
            assert 'Player_Name_N' in self.df.columns, "Required column 'Player_Name_N' not found in DataFrame"
            assert 'Player_Name_E' in self.df.columns, "Required column 'Player_Name_E' not found in DataFrame"
            assert 'Player_Name_S' in self.df.columns, "Required column 'Player_Name_S' not found in DataFrame"
            assert 'Player_Name_W' in self.df.columns, "Required column 'Player_Name_W' not found in DataFrame"
            self.df = self._time_operation(
                "add_pair_names",
                add_pair_names,
                self.df
            )
        
        # Assert column was created
        assert 'Pair_Names_NS' in self.df.columns, "Column 'Pair_Names_NS' was not created"
        assert 'Pair_Names_EW' in self.df.columns, "Column 'Pair_Names_EW' was not created"

    # todo: move this to contract established class
    def _extract_declarer_details(self) -> None:
        # Assert required columns exist
        assert 'Declarer_Direction' in self.df.columns, "Required column 'Declarer_Direction' not found in DataFrame"
        assert 'Player_Name_N' in self.df.columns, "Required column 'Player_Name_N' not found in DataFrame"
        assert 'Player_Name_E' in self.df.columns, "Required column 'Player_Name_E' not found in DataFrame"
        assert 'Player_Name_S' in self.df.columns, "Required column 'Player_Name_S' not found in DataFrame"
        assert 'Player_Name_W' in self.df.columns, "Required column 'Player_Name_W' not found in DataFrame"
        assert 'Player_ID_N' in self.df.columns, "Required column 'Player_ID_N' not found in DataFrame"
        assert 'Player_ID_E' in self.df.columns, "Required column 'Player_ID_E' not found in DataFrame"
        assert 'Player_ID_S' in self.df.columns, "Required column 'Player_ID_S' not found in DataFrame"
        assert 'Player_ID_W' in self.df.columns, "Required column 'Player_ID_W' not found in DataFrame"
        
        self.df = self._time_operation(
            "convert_declarer_columns",
            add_declarer_info,
            self.df
        )
        
        # Assert columns were created
        assert 'Declarer_Name' in self.df.columns, "Column 'Declarer_Name' was not created"
        assert 'Declarer_ID' in self.df.columns, "Column 'Declarer_ID' was not created"

    def _derive_contract_results(self) -> None:
        self.df = self._time_operation("create_result_columns", create_result_columns, self.df)

    def _add_double_dummy_analysis(self) -> None:
        # Assert required columns exist - various contract-related columns needed for DD analysis
        assert 'BidLvl' in self.df.columns, "Required column 'BidLvl' not found in DataFrame"
        assert 'BidSuit' in self.df.columns, "Required column 'BidSuit' not found in DataFrame"
        assert 'Declarer_Direction' in self.df.columns, "Required column 'Declarer_Direction' not found in DataFrame"
        
        self._time_operation("create_dd_tricks_columns", self._create_dd_tricks_columns)
        self._time_operation("create_contract_dependent_dd_scores", self._create_contract_dependent_dd_scores)
        
        # Assert columns were created
        assert 'DD_Tricks' in self.df.columns, "Column 'DD_Tricks' was not created"
        assert 'DD_Tricks_Dummy' in self.df.columns, "Column 'DD_Tricks_Dummy' was not created"
        assert 'DD_Score_NS' in self.df.columns, "Column 'DD_Score_NS' was not created"
        assert 'DD_Score_EW' in self.df.columns, "Column 'DD_Score_EW' was not created"

    def _create_dd_tricks_columns(self) -> None:
        """Create DD_Tricks and DD_Tricks_Dummy columns. DD scores should already exist from DD_SD_Augmenter."""
        # Assert required columns exist
        assert 'BidLvl' in self.df.columns, "Required column 'BidLvl' not found in DataFrame"
        assert 'BidSuit' in self.df.columns, "Required column 'BidSuit' not found in DataFrame"
        assert 'Declarer_Direction' in self.df.columns, "Required column 'Declarer_Direction' not found in DataFrame"
        
        if 'DD_Tricks' not in self.df.columns:
            self.df = self._time_operation("create DD_Tricks", add_dd_tricks_column, self.df)

        if 'DD_Tricks_Dummy' not in self.df.columns:
            self.df = self._time_operation("create DD_Tricks_Dummy", add_dd_tricks_dummy_column, self.df)
            
        # Assert columns were created
        assert 'DD_Tricks' in self.df.columns, "Column 'DD_Tricks' was not created"
        assert 'DD_Tricks_Dummy' in self.df.columns, "Column 'DD_Tricks_Dummy' was not created"

    def _create_contract_dependent_dd_scores(self) -> None:
        """Create contract-dependent DD score columns (DD_Score_Refs, DD_Score_Declarer, DD_Score_NS/EW)."""
        # Assert required columns exist
        assert 'DD_Tricks' in self.df.columns, "Required column 'DD_Tricks' not found in DataFrame"
        assert 'BidLvl' in self.df.columns, "Required column 'BidLvl' not found in DataFrame"
        assert 'BidSuit' in self.df.columns, "Required column 'BidSuit' not found in DataFrame"
        
        self.df = add_dd_scores_contract_dependent(self.df)
        
        # Assert columns were created
        assert 'DD_Score_Declarer' in self.df.columns, "Column 'DD_Score_Declarer' was not created"

    def _compute_expected_values(self) -> None:
        # Assert required columns exist - scores_df and probability columns needed
        # Check for existence of probability columns (sample check)
        prob_cols_exist = any(col.startswith('Probs_') for col in self.df.columns)
        assert prob_cols_exist, "Required probability columns (Probs_*) not found in DataFrame"
        
        self.df = self._time_operation(
            "create_ev_columns",
            add_best_contract_ev,
            self.df,
        )
        
        # Assert EV columns were created
        assert 'EV_Max_NS' in self.df.columns, "Column 'EV_Max_NS' was not created"
        assert 'EV_Max_EW' in self.df.columns, "Column 'EV_Max_EW' was not created"


    def _build_partnership_ev_expressions(self, pd: str) -> List:
        return build_pair_ev_expressions(pd, self.get_vulnerability_conditions)

    def _build_simple_ev_expressions(self, pd: str) -> List:
        return build_ev_expressions_for_pair(pd, self.get_vulnerability_conditions)

    def _build_declarer_specific_ev_expressions(self, pd: str, dd: str) -> List:
        return build_ev_expressions_for_declarer(pd, dd, self.get_vulnerability_conditions)

    def _build_suit_specific_ev_expressions(self, pd: str, dd: str, s: str) -> List:
        return build_ev_expressions_for_strain(pd, dd, s, self.get_vulnerability_conditions)

    def _build_level_specific_ev_expressions(self, pd: str, dd: str, s: str, l: int) -> List:
        return build_ev_expressions_for_level(pd, dd, s, l, self.get_vulnerability_conditions)

    def _add_score_calculations(self) -> None:
        # Assert required columns exist
        assert ('Score_NS' in self.df.columns or 'BidLvl' in self.df.columns), "Required columns for score calculation not found"
        
        self.df = self._time_operation("add_score_calculations", add_score_calculations, self.df)
        
        # Assert columns were created
        assert 'Score' in self.df.columns, "Column 'Score' was not created"
        assert 'Score_NS' in self.df.columns, "Column 'Score_NS' was not created"
        assert 'Score_EW' in self.df.columns, "Column 'Score_EW' was not created"

    def _compute_scoring_differentials(self) -> None:
        # Assert required columns exist
        assert 'Score_NS' in self.df.columns, "Required column 'Score_NS' not found in DataFrame"
        assert 'Score_EW' in self.df.columns, "Required column 'Score_EW' not found in DataFrame"
        assert 'Par_NS' in self.df.columns, "Required column 'Par_NS' not found in DataFrame"
        assert 'Par_EW' in self.df.columns, "Required column 'Par_EW' not found in DataFrame"
        
        self.df = self._time_operation("create_score_diff_columns", create_score_diff_columns, self.df)
        
        # Assert columns were created
        assert 'Par_Diff_NS' in self.df.columns, "Column 'Par_Diff_NS' was not created"
        assert 'Par_Diff_EW' in self.df.columns, "Column 'Par_Diff_EW' was not created"
        assert 'DD_Tricks_Diff' in self.df.columns, "Column 'DD_Tricks_Diff' was not created"

    # todo: would be interesting to enhance this for any contracts and then move into all contract class
    def _apply_law_of_total_tricks(self) -> None:
        # Check if required columns exist - partnership suit lengths needed
        required_columns = [f'SL_{pair}_{suit}' for pair in ['NS', 'EW'] for suit in 'SHDC']
        has_required_columns = all(col in self.df.columns for col in required_columns)
        
        if has_required_columns and 'LoTT_NS' not in self.df.columns:
            self.df = self._time_operation("create LoTT", add_law_of_total_tricks, self.df)
            
            # Assert columns were created
            assert 'LoTT_NS' in self.df.columns, "Column 'LoTT_NS' was not created"
            assert 'LoTT_EW' in self.df.columns, "Column 'LoTT_EW' was not created"
        elif not has_required_columns:
            logger.info("Skipping Law of Total Tricks calculation - required suit length columns not found")

    def _derive_positional_roles(self) -> None:
        # these augmentations should not already exist.
        assert 'Direction_OnLead' not in self.df.columns
        assert 'Defender_Pair_Direction' not in self.df.columns
        assert 'Direction_Dummy' not in self.df.columns
        assert 'OnLead' not in self.df.columns
        assert 'Direction_NotOnLead' not in self.df.columns
        assert 'Dummy' not in self.df.columns
        assert 'Defender_Par_GE' not in self.df.columns
        assert 'EV_Score_Col_Declarer' not in self.df.columns
        assert 'Score_Declarer' not in self.df.columns
        assert 'Par_Declarer' not in self.df.columns
 
        self.df = self._time_operation(
            "create position columns",
            add_position_role_info,
            self.df,
        )

    def _compute_trick_taking_probabilities(self) -> None:
        self.df = self._time_operation(
            "create prob_taking columns",
            add_trick_probabilities,
            self.df,
        )

    def _generate_board_result_metrics(self) -> None:
        self.df = self._time_operation(
            "create board result columns",
            add_declarer_scores,
            self.df,
        )

    def _classify_trick_results(self) -> None:
        self.df = self._time_operation("create trick columns", add_trick_columns, self.df)

    def _incorporate_player_ratings(self) -> None:
        self.df = self._time_operation(
            "create rating columns",
            add_player_ratings,
            self.df,
        )

    def perform_final_contract_augmentations(self) -> pl.DataFrame:
        """Main method to perform final contract augmentations"""
        t_start = time.time()
        logger.info("Starting final contract augmentations")

        self._standardize_contract_format() # 5s
        self._classify_contract_types() # 29s
        self._build_player_ids()
        self._build_player_names()
        self._build_partnership_ids()
        self._build_partnership_names()
        self._extract_declarer_details() # 70s # do sooner?
        self._derive_contract_results()
        self._add_score_calculations()
        self._add_double_dummy_analysis() # 1m30s
        self._compute_expected_values() # 2s
        self._compute_scoring_differentials()
        self._apply_law_of_total_tricks() # 7s # todo: would be interesting to create lott for all contracts and then move into AllContractsAugmenter
        #self._perform_legacy_renames() # 6s
        self._derive_positional_roles() # 1m30s
        self._compute_trick_taking_probabilities() # 7s
        self._generate_board_result_metrics() # 10m
        self._classify_trick_results() # 0s
        self._incorporate_player_ratings() # 3s

        logger.info(f"Final contract augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df
class MatchPointAugmenter:
    def __init__(self, df: pl.DataFrame, incorporate_elo_ratings: bool = False):
        self.df = df
        self.discrete_score_columns = [] # ['DD_Score_NS', 'EV_Max_NS'] # calculate matchpoints for these columns which change with each row's Score_NS
        self.dd_score_columns = DD_SCORE_COLUMNS
        self.ev_score_columns = EV_SCORE_COLUMNS
        self.all_score_columns = self.discrete_score_columns + self.dd_score_columns + self.ev_score_columns
        self.incorporate_elo_ratings = incorporate_elo_ratings

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _compute_matchpoint_top(self) -> None:
        # Assert required columns exist (add_board_matchpoint_top only needs MP_NS and MP_EW)
        assert 'MP_NS' in self.df.columns, "Required column 'MP_NS' not found in DataFrame"
        assert 'MP_EW' in self.df.columns, "Required column 'MP_EW' not found in DataFrame"
        
        if 'MP_Top' not in self.df.columns:
            self.df = self._time_operation("create MP_Top", add_board_matchpoint_top, self.df)
        
        # Assert column was created
        assert 'MP_Top' in self.df.columns, "Column 'MP_Top' was not created"

    def _compute_raw_matchpoints(self) -> None:
        # Assert required columns exist
        assert 'Score_NS' in self.df.columns, "Required column 'Score_NS' not found in DataFrame"
        assert 'Score_EW' in self.df.columns, "Required column 'Score_EW' not found in DataFrame"
        
        # If MP columns are missing, we must compute them and need grouping keys
        if 'MP_NS' not in self.df.columns or 'MP_EW' not in self.df.columns:
            assert 'session_id' in self.df.columns, "Required column 'session_id' not found in DataFrame"
            assert 'Board' in self.df.columns, "Required column 'Board' not found in DataFrame"
            self.df = self._time_operation("calculate matchpoints MP_(NS|EW)", add_matchpoint_scores_from_raw, self.df)
        
        # Assert columns were created
        assert 'MP_NS' in self.df.columns, "Column 'MP_NS' was not created"
        assert 'MP_EW' in self.df.columns, "Column 'MP_EW' was not created"

    def _convert_to_percentages(self) -> None:
        # Assert required columns exist
        assert 'MP_NS' in self.df.columns, "Required column 'MP_NS' not found in DataFrame"
        assert 'MP_EW' in self.df.columns, "Required column 'MP_EW' not found in DataFrame"
        assert 'MP_Top' in self.df.columns, "Required column 'MP_Top' not found in DataFrame"
        
        if 'Pct_NS' not in self.df.columns:
            self.df = self._time_operation("calculate matchpoints percentages", add_percentage_scores, self.df)
        
        # Assert columns were created
        assert 'Pct_NS' in self.df.columns, "Column 'Pct_NS' was not created"
        assert 'Pct_EW' in self.df.columns, "Column 'Pct_EW' was not created"

    def _compute_declarer_percentage(self) -> None:
        # Assert required columns exist
        assert 'Declarer_Pair_Direction' in self.df.columns, "Required column 'Declarer_Pair_Direction' not found in DataFrame"
        assert 'Pct_NS' in self.df.columns, "Required column 'Pct_NS' not found in DataFrame"
        assert 'Pct_EW' in self.df.columns, "Required column 'Pct_EW' not found in DataFrame"
        
        if 'Declarer_Pct' not in self.df.columns:
            self.df = self._time_operation("create Declarer_Pct", add_declarer_percentage, self.df)
        
        # Assert column was created
        assert 'Declarer_Pct' in self.df.columns, "Column 'Declarer_Pct' was not created"

    def _compute_group_matchpoints(self, series_list: list[pl.Series]) -> pl.Series:
        return compute_group_matchpoints(series_list)

    def _compute_comprehensive_matchpoints(self) -> None:
        # Assert required columns exist
        assert 'Score_NS' in self.df.columns, "Required column 'Score_NS' not found in DataFrame"
        assert 'MP_Top' in self.df.columns, "Required column 'MP_Top' not found in DataFrame"
        
        self.df = self._time_operation("compute_all_score_matchpoints", compute_all_score_matchpoints, self.df, self.all_score_columns)
        
        # Assert columns were created (check for at least one MP column)
        mp_columns = [col for col in self.df.columns if col.startswith('MP_') and col != 'MP_Top']
        assert len(mp_columns) > 0, "No MP_* columns were created"
        
        # Post-condition: Ensure DD/Par MP columns were NOT created here (they should be created by specialized functions)
        assert 'MP_DD_Score_NS' not in self.df.columns, "_compute_comprehensive_matchpoints: Incorrectly created 'MP_DD_Score_NS' (should only be created by _convert_dd_scores_to_percentages)"
        assert 'MP_DD_Score_EW' not in self.df.columns, "_compute_comprehensive_matchpoints: Incorrectly created 'MP_DD_Score_EW' (should only be created by _convert_dd_scores_to_percentages)"
        assert 'MP_Par_NS' not in self.df.columns, "_compute_comprehensive_matchpoints: Incorrectly created 'MP_Par_NS' (should only be created by _convert_par_scores_to_percentages)"
        assert 'MP_Par_EW' not in self.df.columns, "_compute_comprehensive_matchpoints: Incorrectly created 'MP_Par_EW' (should only be created by _convert_par_scores_to_percentages)"

    def _compute_mp_percentage_from_score(self, col: str) -> pl.Series:
        """Calculate matchpoint percentage from MP column."""
        return pl.col(f'MP_{col}') / (pl.col('MP_Top') + 1)

    def _compute_partnership_elo_ratings(self) -> None:
        """Calculate Elo pair matchpoint ratings."""
        # Assert required columns exist
        assert 'Pair_Names_NS' in self.df.columns, "Required column 'Pair_Names_NS' not found in DataFrame"
        assert 'Pair_Names_EW' in self.df.columns, "Required column 'Pair_Names_EW' not found in DataFrame"
        assert 'MP_NS' in self.df.columns, "Required column 'MP_NS' not found in DataFrame"
        assert 'MP_EW' in self.df.columns, "Required column 'MP_EW' not found in DataFrame"
        
        self.df = self._time_operation("calculate Elo pair matchpoint ratings", compute_pair_matchpoint_elo_ratings, self.df)
        
        # Assert columns were created
        assert 'Elo_R_NS' in self.df.columns, "Column 'Elo_R_NS' was not created"
        assert 'Elo_R_EW' in self.df.columns, "Column 'Elo_R_EW' was not created"

    def _compute_individual_elo_ratings(self) -> None:
        """Calculate Elo player matchpoint ratings."""
        # Assert required columns exist
        assert 'Player_ID_N' in self.df.columns, "Required column 'Player_ID_N' not found in DataFrame"
        assert 'Player_ID_E' in self.df.columns, "Required column 'Player_ID_E' not found in DataFrame"
        assert 'Player_ID_S' in self.df.columns, "Required column 'Player_ID_S' not found in DataFrame"
        assert 'Player_ID_W' in self.df.columns, "Required column 'Player_ID_W' not found in DataFrame"
        assert 'MP_NS' in self.df.columns, "Required column 'MP_NS' not found in DataFrame"
        assert 'MP_EW' in self.df.columns, "Required column 'MP_EW' not found in DataFrame"
        
        self.df = self._time_operation("calculate Elo player matchpoint ratings", compute_player_matchpoint_elo_ratings, self.df)
        
        # Assert columns were created
        assert 'Elo_R_N' in self.df.columns, "Column 'Elo_R_N' was not created"
        assert 'Elo_R_E' in self.df.columns, "Column 'Elo_R_E' was not created"
        assert 'Elo_R_S' in self.df.columns, "Column 'Elo_R_S' was not created"
        assert 'Elo_R_W' in self.df.columns, "Column 'Elo_R_W' was not created"

    def _track_event_elo_progression(self) -> None:
        """Calculate Elo start/end columns."""
        # Assert required columns exist
        assert 'Elo_R_N' in self.df.columns, "Required column 'Elo_R_N' not found in DataFrame"
        assert 'Elo_R_E' in self.df.columns, "Required column 'Elo_R_E' not found in DataFrame"
        assert 'Elo_R_S' in self.df.columns, "Required column 'Elo_R_S' not found in DataFrame"
        assert 'Elo_R_W' in self.df.columns, "Required column 'Elo_R_W' not found in DataFrame"
        
        self.df = self._time_operation("calculate event start end Elo columns", compute_event_start_end_elo_columns, self.df)
        
        # Assert columns were created
        assert 'Elo_R_N_EventStart' in self.df.columns, "Column 'Elo_R_N_EventStart' was not created"
        assert 'Elo_R_S_EventStart' in self.df.columns, "Column 'Elo_R_S_EventStart' was not created"
        assert 'Elo_R_E_EventStart' in self.df.columns, "Column 'Elo_R_E_EventStart' was not created"
        assert 'Elo_R_W_EventStart' in self.df.columns, "Column 'Elo_R_W_EventStart' was not created"
        assert 'Elo_R_N_EventEnd' in self.df.columns, "Column 'Elo_R_N_EventEnd' was not created"
        assert 'Elo_R_S_EventEnd' in self.df.columns, "Column 'Elo_R_S_EventEnd' was not created"
        assert 'Elo_R_E_EventEnd' in self.df.columns, "Column 'Elo_R_E_EventEnd' was not created"
        assert 'Elo_R_W_EventEnd' in self.df.columns, "Column 'Elo_R_W_EventEnd' was not created"
        assert 'Elo_R_NS_EventStart' in self.df.columns, "Column 'Elo_R_NS_EventStart' was not created"
        assert 'Elo_R_EW_EventStart' in self.df.columns, "Column 'Elo_R_EW_EventStart' was not created"
        assert 'Elo_R_NS_EventEnd' in self.df.columns, "Column 'Elo_R_NS_EventEnd' was not created"
        assert 'Elo_R_EW_EventEnd' in self.df.columns, "Column 'Elo_R_EW_EventEnd' was not created"

    def _finalize_all_scoring_metrics(self) -> None:
        """Calculate final scores and percentages using optimized vectorized operations."""
        self.df = self._time_operation("calculate final scores", self._execute_final_scoring_pipeline, self.df)

    def _execute_final_scoring_pipeline(self, df: pl.DataFrame) -> pl.DataFrame:
        """Internal implementation of final scores calculation."""
        # Split into smaller, more efficient functions
        self._convert_dd_scores_to_percentages()
        self._convert_par_scores_to_percentages()
        self._convert_declarer_scores_to_percentages()
        self._identify_maximum_scores()
        self._calculate_score_differentials()
        
        return self.df

    def _convert_dd_scores_to_percentages(self) -> None:
        """Delegate to global DD score percentage computation."""
        # Pre-conditions: Check required inputs exist
        assert 'Score_NS' in self.df.columns, "_convert_dd_scores_to_percentages: Missing required input 'Score_NS'"
        assert 'Score_EW' in self.df.columns, "_convert_dd_scores_to_percentages: Missing required input 'Score_EW'"
        assert 'DD_Score_NS' in self.df.columns, "_convert_dd_scores_to_percentages: Missing required input 'DD_Score_NS'"
        assert 'DD_Score_EW' in self.df.columns, "_convert_dd_scores_to_percentages: Missing required input 'DD_Score_EW'"
        
        # Pre-conditions: Check outputs don't already exist (prevents duplicate computation)
        assert 'MP_DD_Score_NS' not in self.df.columns, "_convert_dd_scores_to_percentages: Output 'MP_DD_Score_NS' already exists (duplicate computation)"
        assert 'MP_DD_Score_EW' not in self.df.columns, "_convert_dd_scores_to_percentages: Output 'MP_DD_Score_EW' already exists (duplicate computation)"
        assert 'DD_Score_Pct_NS' not in self.df.columns, "_convert_dd_scores_to_percentages: Output 'DD_Score_Pct_NS' already exists (duplicate computation)"
        assert 'DD_Score_Pct_EW' not in self.df.columns, "_convert_dd_scores_to_percentages: Output 'DD_Score_Pct_EW' already exists (duplicate computation)"
        
        self.df = self._time_operation(
            "DD score percentages",
            compute_dd_score_percentages,
            self.df,
        )
        
        # Post-conditions: Verify outputs were created successfully
        assert 'MP_DD_Score_NS' in self.df.columns, "_convert_dd_scores_to_percentages: Failed to create output 'MP_DD_Score_NS'"
        assert 'MP_DD_Score_EW' in self.df.columns, "_convert_dd_scores_to_percentages: Failed to create output 'MP_DD_Score_EW'"
        assert 'DD_Score_Pct_NS' in self.df.columns, "_convert_dd_scores_to_percentages: Failed to create output 'DD_Score_Pct_NS'"
        assert 'DD_Score_Pct_EW' in self.df.columns, "_convert_dd_scores_to_percentages: Failed to create output 'DD_Score_Pct_EW'"

    def _convert_par_scores_to_percentages(self) -> None:
        """Delegate to global Par percentage computation."""
        # Check if Par columns exist before attempting computation
        required_par_cols = ['Par_NS', 'Par_EW']
        if all(col in self.df.columns for col in required_par_cols):
            # Pre-conditions: Check outputs don't already exist (prevents duplicate computation)
            assert 'MP_Par_NS' not in self.df.columns, "_convert_par_scores_to_percentages: Output 'MP_Par_NS' already exists (duplicate computation)"
            assert 'MP_Par_EW' not in self.df.columns, "_convert_par_scores_to_percentages: Output 'MP_Par_EW' already exists (duplicate computation)"
            assert 'Par_Pct_NS' not in self.df.columns, "_convert_par_scores_to_percentages: Output 'Par_Pct_NS' already exists (duplicate computation)"
            assert 'Par_Pct_EW' not in self.df.columns, "_convert_par_scores_to_percentages: Output 'Par_Pct_EW' already exists (duplicate computation)"
            
            self.df = self._time_operation(
                "Par percentages",
                compute_par_percentages,
                self.df,
            )
            
            # Post-conditions: Verify outputs were created successfully
            assert 'MP_Par_NS' in self.df.columns, "_convert_par_scores_to_percentages: Failed to create output 'MP_Par_NS'"
            assert 'MP_Par_EW' in self.df.columns, "_convert_par_scores_to_percentages: Failed to create output 'MP_Par_EW'"
            assert 'Par_Pct_NS' in self.df.columns, "_convert_par_scores_to_percentages: Failed to create output 'Par_Pct_NS'"
            assert 'Par_Pct_EW' in self.df.columns, "_convert_par_scores_to_percentages: Failed to create output 'Par_Pct_EW'"
        else:
            logger.warning("Skipping Par percentage computation - Par columns not found (ParScore likely missing from DD analysis)")

    def _convert_declarer_scores_to_percentages(self) -> None:
        """Delegate to global declarer percentage computation."""
        self.df = self._time_operation(
            "Declarer percentages",
            compute_declarer_percentages,
            self.df,
        )

    def _identify_maximum_scores(self) -> None:
        """Delegate to global max-score computation."""
        self.df = self._time_operation(
            "Max scores",
            compute_max_scores,
            self.df,
        )

    def _calculate_score_differentials(self) -> None:
        """Delegate to global difference-score computation."""
        self.df = self._time_operation(
            "Difference scores",
            compute_score_differences,
            self.df,
        )

    def _incorporate_event_elo_tracking(self) -> None:
        """Delegate to global event start/end Elo computation."""
        self.df = self._time_operation(
            "Event start/end Elo columns",
            compute_event_start_end_elo_columns,
            self.df,
        )

    def perform_matchpoint_augmentations(self) -> pl.DataFrame:
        t_start = time.time()
        logger.info("Starting matchpoint augmentations")
        
        self._compute_raw_matchpoints() # 12s
        self._compute_matchpoint_top() # 5s
        self._convert_to_percentages() # 1s
        self._compute_declarer_percentage() # 1s
        self._compute_comprehensive_matchpoints() # 3m
        self._finalize_all_scoring_metrics() # ?
        if self.incorporate_elo_ratings:
            self._compute_partnership_elo_ratings() # 1s
            self._compute_individual_elo_ratings() # 1s
            self._track_event_elo_progression()

        logger.info(f"Matchpoint augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


# todo: IMP augmentations are not implemented yet
class IMPAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def perform_imp_augmentations(self) -> pl.DataFrame:
        t_start = time.time()
        logger.info("Starting IMP augmentations")
        
        logger.info(f"IMP augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class AllHandRecordAugmentations:
    def __init__(self, df: pl.DataFrame, 
                 hrs_cache_df: Optional[pl.DataFrame] = None, 
                 sd_productions: int = 40, 
                 max_dd_adds: Optional[int] = None,
                 max_sd_adds: Optional[int] = None,
                 output_progress: Optional[bool] = True,
                 progress: Optional[Any] = None,
                 lock_func: Optional[Callable[..., pl.DataFrame]] = None):
        """Initialize the AllAugmentations class with a DataFrame and optional parameters.
        
        Args:
            df: The input DataFrame to augment
            hrs_cache_df: dataframe of cached computes
            sd_productions: Number of single dummy productions to generate
            max_dd_adds: Maximum number of double dummy adds to generate
            max_sd_adds: Maximum number of single dummy adds to generate
            output_progress: Whether to output progress
            progress: Optional progress indicator object
            lock_func: Optional function for thread safety
        """
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_dd_adds = max_dd_adds
        self.max_sd_adds = max_sd_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func

        # instance initialization

        # Double dummy tricks for each player and strain
        dd_cols = {f"DD_{p}_{s}": pl.UInt8 for p in 'NESW' for s in 'CDHSN'}

        # Single dummy probabilities. Note that the order of declarers and suits must match the original schema.
        # The declarer order was found to be N, S, W, E from inspecting the original schema.
        probs_cols = {f"Probs_{pair}_{declarer}_{s}_{i}": pl.Float32 for pair in ['NS', 'EW'] for declarer in 'NESW' for s in 'CDHSN' for i in range(14)}

        # Columns that appear after DD columns and before probability columns
        schema_cols = {
            'PBN': pl.String,
            'Dealer': pl.String,
            'Vul': pl.String,
            **dd_cols,
            'ParScore': pl.Int16,
            'ParNumber': pl.Int8,
            'ParContracts': pl.List(pl.Struct({
                'Level': pl.String, 
                'Strain': pl.String, 
                'Doubled': pl.String, # only '' or 'X' but never 'XX' # todo: make boolean?
                'Pair_Direction': pl.String, 
                'Result': pl.Int16
            })),
            'Probs_Trials': pl.Int64,
            **probs_cols,
        }

        # Convert the dict to a Polars Schema for a valid comparison
        hrs_cache_df_schema = pl.Schema(schema_cols)

        if hrs_cache_df is None:
            self.hrs_cache_df = pl.DataFrame(schema=hrs_cache_df_schema)
        else:
            assert set(self.hrs_cache_df.schema.items()) == set(hrs_cache_df_schema.items()), f"hrs_cache_df schema {self.hrs_cache_df.schema} does not match expected schema {hrs_cache_df_schema}"
        
    def perform_all_hand_record_augmentations(self) -> pl.DataFrame:
        """Execute all hand record augmentation steps. Input is a fully cleaned hand record DataFrame.
        
        Returns:
            The fully augmented hand record DataFrame.
        """
        t_start = time.time()
        logger.info(f"Starting all hand record augmentations on DataFrame with {len(self.df)} rows")
        
        # Step 1: Deal-level augmentations
        deal_augmenter = DealAugmenter(self.df)
        self.df = deal_augmenter.perform_deal_augmentations()
        
        # Step 2: Hand augmentations
        result_augmenter = HandAugmenter(self.df)
        self.df = result_augmenter.perform_hand_augmentations()

        # Step 3: Double dummy and single dummy augmentations
        dd_sd_augmenter = DD_SD_Augmenter(
            self.df, 
            self.hrs_cache_df,  
            self.sd_productions, 
            self.max_dd_adds, 
            self.max_sd_adds, 
            self.output_progress,
            self.progress,
            self.lock_func
        )
        self.df, self.hrs_cache_df = dd_sd_augmenter.perform_dd_sd_augmentations()

        # todo: move this somewhere more sensible.
        # Create Par_NS and Par_EW columns if ParScore exists
        if 'ParScore' in self.df.columns:
            self.df = self.df.with_columns(pl.col('ParScore').alias('Par_NS'))
            self.df = self.df.with_columns(pl.col('ParScore').neg().alias('Par_EW'))
        else:
            logger.warning("ParScore column not found - skipping Par_NS/Par_EW creation")
        
        # todo: move this somewhere more sensible.
        # Create DD columns for pair directions and strains e.g. DD_NS_S
        dd_pair_columns = [
            pl.max_horizontal(f'DD_{pair_direction[0]}_{strain}', f'DD_{pair_direction[1]}_{strain}').alias(f'DD_{pair_direction}_{strain}')
            for pair_direction in ['NS', 'EW']
            for strain in 'SHDCN'
        ]
        self.df = self.df.with_columns(dd_pair_columns)
        
        # Step 4: All contract augmentations
        hand_augmenter = AllContractsAugmenter(self.df)
        self.df = hand_augmenter.perform_all_contracts_augmentations()

        logger.info(f"All hand records augmentations completed in {time.time() - t_start:.2f} seconds")
        
        return self.df, self.hrs_cache_df


class AllBoardResultsAugmentations:
    def __init__(self, df: pl.DataFrame, incorporate_elo_ratings: bool = False):
        self.df = df
        self.incorporate_elo_ratings = incorporate_elo_ratings


    def perform_all_board_results_augmentations(self) -> pl.DataFrame:
        """Execute all board results augmentation steps. Input is a fully augmented hand record DataFrame.
        Only relies on columns within brs_df and not any in hrs_df.

        Returns:
            The fully joined and augmented hand record and board results DataFrame.
        """
        t_start = time.time()
        logger.info(f"Starting all board results augmentations on DataFrame with {len(self.df)} rows")
        
        # Step 5: Final contract augmentations
        hand_augmenter = FinalContractAugmenter(self.df)
        self.df = hand_augmenter.perform_final_contract_augmentations()
        
        # Step 6: Matchpoint augmentations
        matchpoint_augmenter = MatchPointAugmenter(self.df, incorporate_elo_ratings=self.incorporate_elo_ratings)
        self.df = matchpoint_augmenter.perform_matchpoint_augmentations()
        
        # Step 7: IMP augmentations (not implemented yet)
        imp_augmenter = IMPAugmenter(self.df)
        self.df = imp_augmenter.perform_imp_augmentations()
        
        logger.info(f"All board results augmentations completed in {time.time() - t_start:.2f} seconds")
        
        return self.df


class AllAugmentations:
    def __init__(self, df: pl.DataFrame, hrs_cache_df: Optional[pl.DataFrame] = None, sd_productions: int = 40, max_dd_adds: Optional[int] = None, max_sd_adds: Optional[int] = None, output_progress: Optional[bool] = True, progress: Optional[Any] = None, lock_func: Optional[Callable[..., pl.DataFrame]] = None, incorporate_elo_ratings: bool = False):
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_dd_adds = max_dd_adds
        self.max_sd_adds = max_sd_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func
        self.incorporate_elo_ratings = incorporate_elo_ratings

    def perform_all_augmentations(self) -> pl.DataFrame:
        """Execute all augmentation steps.
        
        Returns:
            The fully joined and augmented hand record and board results DataFrame.
        """
        t_start = time.time()
        logger.info(f"Starting all augmentations on DataFrame with {len(self.df)} rows")

        hand_record_augmenter = AllHandRecordAugmentations(self.df, self.hrs_cache_df, self.sd_productions, self.max_dd_adds, self.max_sd_adds, self.output_progress, self.progress, self.lock_func)
        self.df, self.hrs_cache_df = hand_record_augmenter.perform_all_hand_record_augmentations()
        board_results_augmenter = AllBoardResultsAugmentations(self.df, incorporate_elo_ratings=self.incorporate_elo_ratings)
        self.df = board_results_augmenter.perform_all_board_results_augmentations()

        logger.info(f"All augmentations completed in {time.time() - t_start:.2f} seconds")

        return self.df, self.hrs_cache_df