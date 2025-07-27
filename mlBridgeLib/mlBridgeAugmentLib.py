# contains functions to augment df with additional columns
# mostly polars functions

# todo:
# since some columns can be derived from other columns, we should assert that input df has at least on column in each group of derived columns.
# assert that column names don't exist in df.columns for all column creation functions.
# refactor when should *_Dcl columns be created? At end of each func, class, class of it's own?
# if a column already exists, print a message and skip creation.
# if a column already exists, generate a new column and assert that the new column is the same as the existing column.
# print column names and dtypes for all columns generated and skipped.
# print a list of mandatory columns that must be present in df.columns. many columns can be derived from other columns e.g. scoring columns.

import polars as pl
from collections import defaultdict
import sys
import pathlib
from typing import Optional, Union, Callable, Type, Dict, List, Tuple, Any # Added line
import mlBridgeLib.mlBridgeLib as mlBridgeLib
import time

import endplay # for __version__
from endplay.parsers import pbn, lin, json
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
from endplay.dds import calc_dd_table, calc_all_tables, par
from endplay.dealer import generate_deals

from mlBridgeLib.mlBridgeLib import (
    NESW, SHDC, NS_EW,
    PlayerDirectionToPairDirection,
    NextPosition,
    PairDirectionToOpponentPairDirection,
    score
)


def create_hand_nesw_columns(df: pl.DataFrame) -> pl.DataFrame:    
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


def create_hands_lists_column(df: pl.DataFrame) -> pl.DataFrame:
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


def create_suit_nesw_columns(df: pl.DataFrame) -> pl.DataFrame:
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
def OHE_Hands(hands_bin: List[List[Tuple[Optional[str], Optional[str]]]]) -> defaultdict[str, List[Any]]:
    handsbind = defaultdict(list)
    for h in hands_bin:
        for direction,nesw in zip(NESW,h):
            assert nesw[0] is not None and nesw[1] is not None
            handsbind['_'.join(['HB',direction])].append(nesw[0])
    return handsbind


# generic function to augment metrics by suits
def Augment_Metric_By_Suits(metrics: pl.DataFrame, metric: str, dtype: pl.DataType = pl.UInt8) -> pl.DataFrame:
    for d,direction in enumerate(NESW):
        for s,suit in  enumerate(SHDC):
            metrics = metrics.with_columns(
                metrics[metric].map_elements(lambda x: x[1][d][0],return_dtype=dtype).alias('_'.join([metric,direction])),
                metrics[metric].map_elements(lambda x: x[1][d][1][s],return_dtype=dtype).alias('_'.join([metric,direction,suit]))
            )
    for direction in NS_EW:
        metrics = metrics.with_columns((metrics['_'.join([metric,direction[0]])]+metrics['_'.join([metric,direction[1]])]).cast(dtype).alias('_'.join([metric,direction])))
        for s,suit in  enumerate(SHDC):
            metrics = metrics.with_columns((metrics['_'.join([metric,direction[0],suit])]+metrics['_'.join([metric,direction[1],suit])]).cast(dtype).alias('_'.join([metric,direction,suit])))
    return metrics


def update_hrs_cache_df(hrs_cache_df: pl.DataFrame, new_df: pl.DataFrame) -> pl.DataFrame:
    # Print initial row counts
    print(f"hrs_cache_df rows: {hrs_cache_df.height}")
    print(f"new_df rows: {new_df.height}")
    
    # Calculate which rows will be added vs replaced
    existing_pbns = set(hrs_cache_df['PBN'].to_list())
    new_pbns = set(new_df['PBN'].to_list())
    
    pbns_to_replace = existing_pbns & new_pbns  # intersection
    pbns_to_add = new_pbns - existing_pbns      # difference
    
    print(f"Rows to be replaced: {len(pbns_to_replace)}")
    print(f"Rows to be added: {len(pbns_to_add)}")
    print(f"Expected final row count: {hrs_cache_df.height + len(pbns_to_add)}")
    
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
        print(f"Added {len(missing_columns)} missing columns to {new_rows.height} new rows")
    
    print(f"Final hrs_cache_df rows: {hrs_cache_df.height}")
    print(f"Net rows added: {len(pbns_to_add)}")
    
    return hrs_cache_df


# calculate dict of contract result scores. each column contains (non-vul,vul) scores for each trick taken. sets are always penalty doubled.
def calculate_scores() -> Tuple[Dict[Tuple, int], Dict[Tuple, int], pl.DataFrame]:

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

    # create score dataframe from dict
    sd = defaultdict(list)
    for suit in 'SHDCN':
        for level in range(1,8):
            for i in range(14):
                sd['_'.join(['Score',str(level)+suit])].append([scores_d[(level,suit,i,False)],scores_d[(level,suit,i,True)]])
    scores_df = pl.DataFrame(sd,orient='row')
    return all_scores_d, scores_d, scores_df


def display_double_dummy_deals(deals: List[Deal], dd_result_tables: List[Any], deal_index: int = 0, max_display: int = 4) -> None:
    # Display a few hands and double dummy tables
    for dd, rt in zip(deals[deal_index:deal_index+max_display], dd_result_tables[deal_index:deal_index+max_display]):
        deal_index += 1
        print(f"Deal: {deal_index}")
        print(dd)
        rt.pprint()


# todo: could save a couple seconds by creating dict of deals
def calc_double_dummy_deals(deals: List[Deal], batch_size: int = 40, output_progress: bool = False, progress: Optional[Any] = None) -> List[Any]:
    # was the wonkyness due to unique() not having maintain_order=True? Let's see if it behaves now.
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
                    print(f"{percent_complete}%: Double dummies calculated for {b} of {len(deals)} unique deals.")
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
            print(f"100%: Double dummies calculated for {len(deals)} unique deals.")
    return all_result_tables


# takes 10000/hour
# easier to combine calculations of double dummy and par scores into one function. Otherwise, we would need to calculate par scores from columns.
def calculate_ddtricks_par_scores(hrs_df: pl.DataFrame, hrs_cache_df: pl.DataFrame, max_adds: Optional[int] = None, output_progress: bool = True, progress: Optional[Any] = None) -> pl.DataFrame:

    # Calculate double dummy and par
    print(f"{hrs_df.height=}")
    print(f"{hrs_cache_df.height=}")
    assert hrs_df['PBN'].null_count() == 0, "PBNs in df must be non-null"
    assert hrs_df.filter(pl.col('PBN').str.len_chars().ne(69)).height == 0, hrs_df.filter(pl.col('PBN').str.len_chars().ne(69))
    assert hrs_cache_df['PBN'].null_count() == 0, "PBNs in hrs_cache_df must be non-null"
    assert hrs_cache_df.filter(pl.col('PBN').str.len_chars().ne(69)).height == 0, hrs_cache_df.filter(pl.col('PBN').str.len_chars().ne(69))
    unique_hrs_df_pbns = set(hrs_df['PBN']) # could be non-unique PBN with difference Dealer, Vul.
    print(f"{len(unique_hrs_df_pbns)=}")
    hrs_cache_with_nulls_df = hrs_cache_df.filter(pl.col('DD_N_C').is_null() | pl.col('ParScore').is_null())
    print(f"{len(hrs_cache_with_nulls_df)=}")
    hrs_cache_with_nulls_pbns = hrs_cache_with_nulls_df['PBN']
    print(f"{len(hrs_cache_with_nulls_pbns)=}")
    unique_hrs_cache_with_nulls_pbns = set(hrs_cache_with_nulls_pbns)
    print(f"{len(unique_hrs_cache_with_nulls_pbns)=}")
    
    hrs_cache_all_pbns = set(hrs_cache_df['PBN'])
    pbns_to_add = set(unique_hrs_df_pbns) - hrs_cache_all_pbns  # In hrs_df but NOT in hrs_cache_df
    print(f"{len(pbns_to_add)=}")
    pbns_to_replace = set(unique_hrs_df_pbns).intersection(set(unique_hrs_cache_with_nulls_pbns))  # In both, with nulls in hrs_cache_df
    print(f"{len(pbns_to_replace)=}")
    pbns_to_process = pbns_to_add.union(pbns_to_replace)
    print(f"{len(pbns_to_process)=}")
    
    if max_adds is not None:
        pbns_to_process = list(pbns_to_process)[:max_adds]
        print(f"limit: {max_adds=} {len(pbns_to_process)=}")
    
    cleaned_pbns = [Deal(pbn) for pbn in pbns_to_process]
    assert all([pbn == dpbn.to_pbn() for pbn,dpbn in zip(pbns_to_process,cleaned_pbns)]), [(pbn,dpbn.to_pbn()) for pbn,dpbn in zip(pbns_to_process,cleaned_pbns) if pbn != dpbn.to_pbn()] # usually a sort order issue which should have been fixed in previous step
    unique_dd_tables = calc_double_dummy_deals(cleaned_pbns, output_progress=output_progress, progress=progress)
    print(f"{len(unique_dd_tables)=}")
    unique_dd_tables_d = {deal.to_pbn():rt for deal,rt in zip(cleaned_pbns,unique_dd_tables)}
    print(f"{len(unique_dd_tables_d)=}")

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

    # Create dataframe of par scores using double dummy
    d = defaultdict(list)
    dd_columns = {f'DD_{direction}_{suit}':pl.UInt8 for suit in 'SHDCN' for direction in 'NESW'}

    # Get Dealer/Vul from appropriate source (either hrs_df or hrs_cache_df) for each PBN type
    source_rows = []
    if pbns_to_add:
        source_rows.extend(hrs_df.filter(pl.col('PBN').is_in(list(pbns_to_add)))[['PBN','Dealer','Vul']].unique().rows())
    if pbns_to_replace:
        source_rows.extend(hrs_cache_df.filter(pl.col('PBN').is_in(list(pbns_to_replace)))[['PBN','Dealer','Vul']].unique().rows())
    print(f"pbns_to_add: {len(pbns_to_add)=}")
    print(f"pbns_to_replace: {len(pbns_to_replace)=}")
    print(f"source_rows: {len(source_rows)=}")
    for pbn, dealer, vul in source_rows:
        if pbn not in unique_dd_tables_d:
            continue
        dd_rows = sum(unique_dd_tables_d[pbn].to_list(), []) # flatten dd_table
        d['PBN'].append(pbn)
        for col,dd in zip(dd_columns, dd_rows):
            d[col].append(dd)
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
    print(f"filtered_schema: {filtered_schema}")
    error_filtered_schema = {k: None for k, v in d.items() if k not in hrs_cache_df.columns}
    print(f"error_filtered_schema: {error_filtered_schema}")
    assert len(error_filtered_schema) == 0, f"error_filtered_schema: {error_filtered_schema}"
    dd_par_df = pl.DataFrame(d, schema=filtered_schema)
    return dd_par_df


def constraints(deal: Deal) -> bool:
    return True


def generate_single_dummy_deals(predeal_string: str, produce: int, env: Dict[str, Any] = dict(), max_attempts: int = 1000000, seed: int = 42, show_progress: bool = True, strict: bool = True, swapping: int = 0) -> Tuple[Tuple[Deal, ...], List[Any]]:
    
    predeal = Deal(predeal_string)

    deals_t = generate_deals(
        constraints,
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
    
    return deals, calc_double_dummy_deals(deals)


def calculate_single_dummy_probabilities(deal: str, produce: int = 100) -> Tuple[Dict[str, pl.DataFrame], Tuple[int, Dict[Tuple[str, str, str], List[float]]]]:

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
        #print(f"predeal:{predeal_string}")

        sd_deals, sd_dd_result_tables = generate_single_dummy_deals(predeal_string, produce, show_progress=False)

        #display_double_dummy_deals(sd_deals, sd_dd_result_tables, 0, 4)
        SD_Tricks_df[ns_ew] = pl.DataFrame([[sddeal.to_pbn()]+[s for d in t.to_list() for s in d] for sddeal,t in zip(sd_deals,sd_dd_result_tables)],schema={'SD_Deal':pl.String}|{'_'.join(['SD_Tricks',d,s]):pl.UInt8 for s in 'SHDCN' for d in 'NESW'},orient='row')

        for d in 'NESW':
            for s in 'SHDCN':
                # always create 14 rows (0-13 tricks taken) for combo of direction and suit. fill never-happened with proper index and 0.0 prob value.
                #ns_ew_rows[(ns_ew,d,s)] = dd_df[d+s].to_pandas().value_counts(normalize=True).reindex(range(14), fill_value=0).tolist() # ['Fixed_Direction','Declarer_Direction','Suit']+['SD_Prob_Take_'+str(n) for n in range(14)]
                vc = {ds:p for ds,p in SD_Tricks_df[ns_ew]['_'.join(['SD_Tricks',d,s])].value_counts(normalize=True).rows()}
                index = {i:0.0 for i in range(14)} # fill values for missing probs
                ns_ew_rows[(ns_ew,d,s)] = list((index|vc).values())

    return SD_Tricks_df, (produce, ns_ew_rows)


# def append_single_dummy_results(pbns,sd_cache_d,produce=100):
#     for pbn in pbns:
#         if pbn not in sd_cache_d:
#             sd_cache_d[pbn] = calculate_single_dummy_probabilities(pbn, produce) # all combinations of declarer pair directI. ion, declarer direciton, suit, tricks taken
#     return sd_cache_d


# performs at 10000/hr
def calculate_sd_probs(df: pl.DataFrame, hrs_cache_df: pl.DataFrame, sd_productions: int = 100, max_adds=None, progress: Optional[Any] = None) -> Tuple[Dict[str, pl.DataFrame], pl.DataFrame]:

    # calculate single dummy probabilities. if already calculated use cache value else update e with new result.
    sd_d = {}
    sd_dfs_d = {}
    assert hrs_cache_df.height == hrs_cache_df['PBN'].n_unique(), "PBNs in hrs_cache_df must be unique"
    pbns_to_add = set(df['PBN'])-set(hrs_cache_df['PBN'])
    print(f"{len(pbns_to_add)=}")
    pbns_to_replace = set(hrs_cache_df.filter(pl.col('PBN').is_in(df['PBN'].to_list()) & pl.col('Probs_Trials').is_null())['PBN'].to_list())
    print(f"{len(pbns_to_replace)=}")
    assert hrs_cache_df.filter(pl.col('PBN').is_in(pbns_to_replace) & pl.col('Probs_Trials').is_null()).height == len(pbns_to_replace), "PBN not a valid replacement"
    pbns_to_process = pbns_to_add.union(pbns_to_replace)
    print(f"{len(pbns_to_process)=}")
    if max_adds is not None:
        pbns_to_process = list(pbns_to_process)[:max_adds]
        print(f"limit: {max_adds=} {len(pbns_to_process)=}")
    cleaned_pbns = [Deal(pbn) for pbn in pbns_to_process]
    assert all([pbn == dpbn.to_pbn() for pbn,dpbn in zip(pbns_to_process,cleaned_pbns)]), [(pbn,dpbn.to_pbn()) for pbn,dpbn in zip(pbns_to_process,cleaned_pbns) if pbn != dpbn.to_pbn()] # usually a sort order issue which should have been fixed in previous step
    print(f"processing time assuming 10000/hour:{len(pbns_to_process)/10000} hours")
    for i,pbn in enumerate(pbns_to_process):
        if progress:
            percent_complete = int(i*100/len(pbns_to_process))
            if hasattr(progress, 'progress'): # streamlit
                progress.progress(percent_complete, f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal. This step takes 30 seconds...")
            elif hasattr(progress, 'set_description'): # tqdm
                progress.set_description(f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal. This step takes 30 seconds...")
        else:
            if i < 10 or i % 10000 == 0:
                percent_complete = int(i*100/len(pbns_to_process))
                print(f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
        if not progress and (i < 10 or i % 10000 == 0):
            t = time.time()
        sd_dfs_d[pbn], sd_d[pbn] = calculate_single_dummy_probabilities(pbn, sd_productions) # all combinations of declarer pair direction, declarer direciton, suit, tricks taken
        if not progress and (i < 10 or i % 10000 == 0):
            print(f"calculate_single_dummy_probabilities: time:{time.time()-t} seconds")
        #error
    if progress:
        if hasattr(progress, 'progress'): # streamlit
            progress.progress(100, f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
        elif hasattr(progress, 'set_description'): # tqdm
            progress.set_description(f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
    else:
        print(f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")

    # create single dummy trick taking probability distribution columns
    sd_probs_d = defaultdict(list)
    for pbn, v in sd_d.items():
        productions, probs_d = v
        sd_probs_d['PBN'].append(pbn)
        sd_probs_d['Probs_Trials'].append(productions)
        for (pair_direction,declarer_direction,suit),probs in probs_d.items():
            #print(pair_direction,declarer_direction,suit)
            for i,t in enumerate(probs):
                sd_probs_d['_'.join(['Probs',pair_direction,declarer_direction,suit,str(i)])].append(t)
    # st.write(sd_probs_d)
    sd_probs_df = pl.DataFrame(sd_probs_d,orient='row')

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


def create_scores_df_with_vul(scores_df: pl.DataFrame) -> pl.DataFrame:
    # Pre-compute score columns
    score_columns = {f'Score_{level}{suit}': scores_df[f'Score_{level}{suit}']
                    for level in range(1, 8) for suit in 'CDHSN'}

    # Create a DataFrame from the score_columns dictionary
    df_scores = pl.DataFrame(score_columns)

    # Explode each column into two separate columns
    exploded_columns = []
    for col in df_scores.columns:
        exploded_columns.extend([
            pl.col(col).list.get(0).alias(f"{col}_NV"),  # Non-vulnerable score
            pl.col(col).list.get(1).alias(f"{col}_V")    # Vulnerable score
        ])

    return df_scores.with_columns(exploded_columns).drop(df_scores.columns)


# def get_cached_sd_data(pbn: str, hrs_d: Dict[str, Any]) -> Dict[str, Union[str, float]]:
#     sd_data = hrs_d[pbn]['SD'][1]
#     row_data = {'PBN': pbn}
#     for (pair_direction, declarer_direction, strain), probs in sd_data.items():
#         col_prefix = f"{pair_direction}_{declarer_direction}_{strain}"
#         for i, prob in enumerate(probs):
#             row_data[f"{col_prefix}_{i}"] = prob
#     return row_data


def calculate_sd_expected_values(df: pl.DataFrame, scores_df: pl.DataFrame) -> pl.DataFrame:

    # retrieve probabilities from cache
    #sd_probs = [get_cached_sd_data(pbn, hrs_d) for pbn in df['PBN']]

    # Create a DataFrame from the extracted sd probs (frequency distribution of tricks).
    #sd_df = pl.DataFrame(sd_probs)

    # todo: look for other places where this is called. duplicated code?
    scores_df_vuls = create_scores_df_with_vul(scores_df)

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
    #print("Results with prob*score:")
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

    #print("\nResults with expected value:")
    return df


# Function to create columns of max values from various regexes of columns. also creates columns of the column names of the max value.
def max_horizontal_and_col(df, pattern):
    cols = df.select(pl.col(pattern)).columns
    max_expr = pl.max_horizontal(pl.col(pattern))
    col_expr = pl.when(pl.col(cols[0]) == max_expr).then(pl.lit(cols[0]))
    for col in cols[1:]:
        col_expr = col_expr.when(pl.col(col) == max_expr).then(pl.lit(col))
    return max_expr, col_expr.otherwise(pl.lit(""))


# calculate EV max scores for various regexes including all vulnerabilities. also create columns of the column names of the max values.
def create_best_contracts(df: pl.DataFrame) -> pl.DataFrame:

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
        max_expr, col_expr = max_horizontal_and_col(df, ev_columns)
        max_ev_dict[f'EV_{v}_Max'] = max_expr
        max_ev_dict[f'EV_{v}_Max_Col'] = col_expr

        for pd in pair_directions:
            # Level 3: Max EV for each pair direction and vulnerability
            ev_columns = f'^EV_{pd}_[NESW]_[SHDCN]_[1-7]_{v}$'
            max_expr, col_expr = max_horizontal_and_col(df, ev_columns)
            max_ev_dict[f'EV_{pd}_{v}_Max'] = max_expr
            max_ev_dict[f'EV_{pd}_{v}_Max_Col'] = col_expr

            for dd in pd: #declarer_directions:
                # Level 2: Max EV for each pair direction, declarer direction, and vulnerability
                ev_columns = f'^EV_{pd}_{dd}_[SHDCN]_[1-7]_{v}$'
                max_expr, col_expr = max_horizontal_and_col(df, ev_columns)
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max'] = max_expr
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max_Col'] = col_expr

                for s in strains:
                    # Level 1: Max EV for each combination
                    ev_columns = f'^EV_{pd}_{dd}_{s}_[1-7]_{v}$'
                    max_expr, col_expr = max_horizontal_and_col(df, ev_columns)
                    max_ev_dict[f'EV_{pd}_{dd}_{s}_{v}_Max'] = max_expr
                    max_ev_dict[f'EV_{pd}_{dd}_{s}_{v}_Max_Col'] = col_expr

    # Create expressions list from dictionary
    t = time.time()
    all_max_ev_expr = [expr.alias(alias) for alias, expr in max_ev_dict.items()]
    print(f"create_best_contracts: all_max_ev_expr created: time:{time.time()-t} seconds")

    # Create a new DataFrame with only the new columns
    # todo: this step is inexplicably slow. appears to take 6 seconds regardless of row count?
    t = time.time()
    df = df.select(all_max_ev_expr)
    print(f"create_best_contracts: sd_ev_max_df created: time:{time.time()-t} seconds")

    return df


def convert_contract_to_contract(df: pl.DataFrame) -> pl.Series:
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
def convert_contract_to_declarer(df: pl.DataFrame) -> List[Optional[str]]:
    return [None if c is None or c == 'PASS' else c[-1] for c in df['Contract']] # extract declarer from contract


def convert_contract_to_pair_declarer(df: pl.DataFrame) -> List[Optional[str]]:
    return [None if c is None or c == 'PASS' else 'NS' if c[-1] in 'NS' else 'EW' for c in df['Contract']] # extract declarer from contract


def convert_contract_to_vul_declarer(df: pl.DataFrame) -> List[Optional[str]]:
    return [None if c is None or c == 'PASS' else bool(v&1) if c[-1] in 'NS' else bool(v&2) for c,v in zip(df['Contract'],df['iVul'])] # extract declarer from contract


def convert_contract_to_level(df: pl.DataFrame) -> List[Optional[int]]:
    return [None if c is None or c == 'PASS' else int(c[0]) for c in df['Contract']] # extract level from contract


def convert_contract_to_strain(df: pl.DataFrame) -> List[Optional[str]]:
    return [None if c is None or c == 'PASS' else c[1] for c in df['Contract']] # extract strain from contract


def convert_contract_to_dbl(df: pl.DataFrame) -> List[Optional[str]]:
    return [None if c is None or c == 'PASS' else c[2:-1] for c in df['Contract']] # extract dbl from contract


def convert_declarer_to_DeclarerName(df: pl.DataFrame) -> List[Optional[str]]:
    return [None if d is None else df[d][i] for i,d in enumerate(df['Declarer_Direction'])] # extract declarer name using declarer direction as the lookup key


def convert_declarer_to_DeclarerID(df: pl.DataFrame) -> List[Optional[str]]:
    return [None if d is None else df[f'Player_ID_{d}'][i] for i,d in enumerate(df['Declarer_Direction'])] # extract declarer name using declarer direction as the lookup key


# convert to ml df needs to perform this.
def convert_contract_to_result(df: pl.DataFrame) -> List[Optional[int]]:
    assert False, "convert_contract_to_result not implemented. Must be done in convert_to_mldf()."
#    return [None if c is None or c == 'PASS' else 0 if c[-1] in ['=','0'] else int(c[-1]) if c[-2] == '+' else -int(c[-1]) for c in df['Contract']] # create result from contract


def convert_tricks_to_result(df: pl.DataFrame) -> List[Optional[int]]:
    return [None if t is None or c == 'PASS' else t-6-int(c[0]) for c,t in zip(df['Contract'],df['Tricks'])] # create result from contracts and tricks


def convert_contract_to_tricks(df: pl.DataFrame) -> List[Optional[int]]:
    return [None if c is None or c == 'PASS' or r is None else int(c[0])+6+r for c,r in zip(df['Contract'],df['Result'])] # create tricks from contract and result


def convert_contract_to_DD_Tricks(df: pl.DataFrame) -> List[Optional[int]]:
    return [None if c is None or c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Declarer_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


def convert_contract_to_DD_Tricks_Dummy(df: pl.DataFrame) -> List[Optional[int]]:
    return [None if c is None or c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Dummy_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


def convert_contract_to_DD_Score_Ref(df: pl.DataFrame) -> pl.DataFrame:
    # create DD_Score_Refs which contains the name of the DD_Score_[1-7][CDHSN]_[NESW] column which contains the double dummy score for the contract.
    # could use pl.str_concat() instead
    df = df.with_columns(
        (pl.lit('DD_Score_')+pl.col('BidLvl').cast(pl.String)+pl.col('BidSuit')+pl.lit('_')+pl.col('Declarer_Direction')).alias('DD_Score_Refs'),
    )
    all_scores_d, scores_d, scores_df = calculate_scores()
    # Create scores for columns: DD_Score_[1-7][CDHSN]_[NESW]. Calculated in respect to the player's direction. e.g. DD_Score_1N_E is the score given E as the declarer.
    df = df.with_columns([
        pl.struct([f"DD_{direction}_{strain}", f"Vul_{pair_direction}"]) # todo: change Vul_{pair_direction} to use iVul so brs_df can be used without joining Vul_(NS|EW).
        .map_elements(
            lambda r, lvl=level, strn=strain, dir=direction, pdir=pair_direction: 
                scores_d.get((lvl, strn, r[f"DD_{dir}_{strn}"], r[f"Vul_{pdir}"]), 0), # default becomes 0. ok? should only occur in the case of null (PASS).
            return_dtype=pl.Int16
        )
        .alias(f"DD_Score_{level}{strain}_{direction}")
        for level in range(1, 8)
        for strain in mlBridgeLib.CDHSN
        for direction, pair_direction in [('N','NS'), ('E','EW'), ('S','NS'), ('W','EW')]
    ])

    # Create list of column names: DD_Score_[1-7][CDHSN]_[NESW]
    dd_score_columns = [f"DD_Score_{level}{strain}_{direction}" 
                        for level in range(1, 8)
                        for strain in mlBridgeLib.CDHSN  
                        for direction in mlBridgeLib.NESW]
    # Create DD_Score_Declarer by selecting the DD_Score_[1-7][CDHSN]_[NESW] column for the given Declarer_Direction.
    df = df.with_columns([
        pl.struct(['BidLvl', 'BidSuit', 'Declarer_Direction'] + dd_score_columns)
        .map_elements(
            lambda r: None if r['Declarer_Direction'] is None else r[f"DD_Score_{r['BidLvl']}{r['BidSuit']}_{r['Declarer_Direction']}"],
            return_dtype=pl.Int16
        )
        .alias('DD_Score_Declarer')
    ])

    df = df.with_columns(
        pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
            .then(pl.col('DD_Score_Declarer'))
            .when(pl.col('Declarer_Pair_Direction').eq('EW'))
            .then(pl.col('DD_Score_Declarer').neg())
            .otherwise(0) # we want PASS to have a score of 0, right?
            .alias('DD_Score_NS'),
        
        pl.when(pl.col('Declarer_Pair_Direction').eq('EW'))
            .then(pl.col('DD_Score_Declarer'))
            .when(pl.col('Declarer_Pair_Direction').eq('NS'))
            .then(pl.col('DD_Score_Declarer').neg())
            .otherwise(0) # we want PASS to have a score of 0, right?
            .alias('DD_Score_EW')
    )
    return df

# todo: implement this
def AugmentACBLHandRecords(df: pl.DataFrame) -> pl.DataFrame:

    augmenter = FinalContractAugmenter(df)
    df = augmenter.perform_final_contract_augmentations()

    # takes 5s
    if 'game_date' in df.columns:
        t = time.time()
        df = df.with_columns(pl.Series('Date',df['game_date'].str.strptime(pl.Date,'%Y-%m-%d %H:%M:%S')))
        print(f"Time to create ACBL Date: {time.time()-t} seconds")
    # takes 5s
    if 'hand_record_id' in df.columns:
        t = time.time()
        df = df.with_columns(
            pl.col('hand_record_id').cast(pl.String),
        )
        print(f"Time to create ACBL hand_record_id: {time.time()-t} seconds")
    return df


def Perform_Legacy_Renames(df: pl.DataFrame) -> pl.DataFrame:

    df = df.with_columns([
        #pl.col('Section').alias('section_name'), # will this be needed for acbl?
        pl.col('N').alias('Player_Name_N'),
        pl.col('S').alias('Player_Name_S'),
        pl.col('E').alias('Player_Name_E'),
        pl.col('W').alias('Player_Name_W'),
        pl.col('Declarer_Name').alias('Name_Declarer'),
        pl.col('Declarer_ID').alias('Number_Declarer'), #  todo: rename to 'Declarer_ID'?
        pl.concat_list(['N', 'S']).alias('Player_Names_NS'),
        pl.concat_list(['E', 'W']).alias('Player_Names_EW'),
        # EV legacy renames
        # pl.col('EV_Max_Col').alias('SD_Contract_Max'), # Pair direction invariant.
        # pl.col('EV_Max_NS').alias('SD_Score_NS'),
        # pl.col('EV_Max_EW').alias('SD_Score_EW'),
        # pl.col('EV_Max_NS').alias('SD_Score_Max_NS'),
        # pl.col('EV_Max_EW').alias('SD_Score_Max_EW'),
        # (pl.col('EV_Max_NS')-pl.col('Score_NS')).alias('SD_Score_Diff_NS'),
        # (pl.col('EV_Max_EW')-pl.col('Score_EW')).alias('SD_Score_Diff_EW'),
        # (pl.col('EV_Max_NS')-pl.col('Score_NS')).alias('SD_Score_Max_Diff_NS'),
        # (pl.col('EV_Max_EW')-pl.col('Score_EW')).alias('SD_Score_Max_Diff_EW'),
        # (pl.col('EV_Max_NS')-pl.col('Pct_NS')).alias('SD_Pct_Diff_NS'),
        # (pl.col('EV_Max_EW')-pl.col('Pct_EW')).alias('SD_Pct_Diff_EW'),
        ])
    return df


def DealToCards(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col(f'Suit_{direction}_{suit}').str.contains(rank).alias(f'C_{direction}{suit}{rank}')
        for direction in 'NESW'
        for suit in 'SHDC'
        for rank in 'AKQJT98765432'
    ])
    return df


def CardsToHCP(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate High Card Points (HCP) for a bridge hand dataset.
    
    Args:
    df (pl.DataFrame): Input DataFrame with columns named C_{direction}{suit}{rank}
                       where direction is N, E, S, W, suit is S, H, D, C, and rank is A, K, Q, J.
    
    Returns:
    pl.DataFrame: Input DataFrame with additional HCP columns.
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


def CardsToQuickTricks(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate Quick Tricks for a bridge hand dataset.
    
    Args:
    df (pl.DataFrame): Input DataFrame with Suit_{direction}_{suit} columns.
    
    Returns:
    pl.DataFrame: DataFrame with additional Quick Tricks columns.
    """
    qt_dict = {'AK': 2.0, 'AQ': 1.5, 'A': 1.0, 'KQ': 1.0, 'K': 0.5}
    
    # Calculate QT for each suit
    qt_expr = [
        pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('AK')).then(pl.lit(2.0))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('AQ')).then(pl.lit(1.5))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('A')).then(pl.lit(1.0))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('KQ')).then(pl.lit(1.0))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('K')).then(pl.lit(0.5))
        .otherwise(pl.lit(0.0)).alias(f'QT_{d}_{s}')
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


def calculate_LoTT(df: pl.DataFrame) -> pl.DataFrame:

    for max_col in ['SL_Max_NS','SL_Max_EW']:
        if max_col not in df.columns:
            raise ValueError(f"The DataFrame must contain the '{max_col}' column")
    
        # Get unique values from SL_Max_(NS|EW) columns
        sl_max_columns = df[max_col].unique(maintain_order=True).to_list()

        print(f"Unique {max_col} columns:", sl_max_columns)
        
        # Create SL columns of either 0 or the value of the row in SL_Max_(NS|EW)
        sl_columns = [
            pl.when(pl.col(max_col) == col)
            .then(pl.col(col))
            .otherwise(0).alias(f"LoTT_{col}") # LoTT_{SL_(NS|EW)_[SHDC]}
            for col in sl_max_columns
        ]
        
        # Create DD columns of either 0 or the value of the row in SL_Max_(NS|EW) -> DD_(NS|EW)_[SHDC].
        dd_columns = [
            pl.when(pl.col(max_col) == col)
            .then(pl.col(f"DD_{col[-4:]}")) # DD_{(NS|EW)_[SHDC]}
            .otherwise(0).alias(f"LoTT_DD_{col[-4:]}") # LoTT_DD_{(NS|EW)_[SHDC]}
            for col in sl_max_columns
        ]
        
        # Add SL_(NS|EW)_[SHDC] columns and DD_(NS|EW)_[SHDC] columns to df.
        df = df.with_columns(sl_columns+dd_columns)
        #print(df)
        
        # Sum horizontally LoTT_SL_{(NS|EW)}_[SHDC] columns and LoTT_DD_{(NS|EW)}_[SHDC] columns.
        df = df.with_columns([
            pl.sum_horizontal(pl.col(f'^LoTT_SL_{max_col[-2:]}_[SHDC]$')).alias(f'LoTT_SL_{max_col[-2:]}'),
            pl.sum_horizontal(pl.col(f'^LoTT_DD_{max_col[-2:]}_[SHDC]$')).alias(f'LoTT_DD_{max_col[-2:]}'),
        ])

    # Sum LoTT_SL_(NS|EW) columns and LoTT_DD_(NS|EW) columns.
    df = df.with_columns([
        pl.sum_horizontal(pl.col(r'^LoTT_SL_(NS|EW)$')).alias('LoTT_SL'),
        pl.sum_horizontal(pl.col(r'^LoTT_DD_(NS|EW)$')).alias('LoTT_DD')
    ])
    df = df.with_columns((pl.col('LoTT_SL')-pl.col('LoTT_DD').cast(pl.Int8)).alias('LoTT_Diff'))
    
    return df


class DealAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _add_default_columns(self) -> None:
        if 'group_id' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit(0).alias('group_id'))
        if 'session_id' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit(0).alias('session_id'))
        if 'section_name' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit('').alias('section_name'))

    def _create_dealer(self) -> None:
        if 'Dealer' not in self.df.columns:
            def board_number_to_dealer(bn):
                return 'NESW'[(bn-1) & 3]
            
            self.df = self._time_operation(
                "create Dealer",
                lambda df: df.with_columns(
                    pl.col('board_number')
                    .map_elements(board_number_to_dealer, return_dtype=pl.String)
                    .alias('Dealer')
                ),
                self.df
            )

    def _create_vulnerability(self) -> None:
        if 'iVul' not in self.df.columns:
            if 'Vul' in self.df.columns:
                def vul_to_ivul(vul: str) -> int:
                    return ['None','N_S','E_W','Both'].index(vul)
                
                self.df = self._time_operation(
                    "create iVul from Vul",
                    lambda df: df.with_columns(
                        pl.col('Vul')
                        .map_elements(vul_to_ivul, return_dtype=pl.UInt8)
                        .alias('iVul')
                    ),
                    self.df
                )
            else:
                def board_number_to_vul(bn: int) -> int:
                    bn -= 1
                    return range(bn//4, bn//4+4)[bn & 3] & 3
                
                self.df = self._time_operation(
                    "create iVul from Board",
                    lambda df: df.with_columns(
                        pl.col('Board')
                        .map_elements(board_number_to_vul, return_dtype=pl.UInt8)
                        .alias('iVul')
                    ),
                    self.df
                )

        if 'Vul' not in self.df.columns:
            def ivul_to_vul(ivul: int) -> str:
                return ['None','N_S','E_W','Both'][ivul]
            
            self.df = self._time_operation(
                "create Vul from iVul",
                lambda df: df.with_columns(
                    pl.col('iVul')
                    .map_elements(ivul_to_vul, return_dtype=pl.String)
                    .alias('Vul')
                ),
                self.df
            )

        if 'Vul_NS' not in self.df.columns:
            self.df = self._time_operation(
                "create Vul_NS/EW",
                lambda df: df.with_columns([
                    pl.Series('Vul_NS', df['Vul'].is_in(['N_S','Both']), pl.Boolean),
                    pl.Series('Vul_EW', df['Vul'].is_in(['E_W','Both']), pl.Boolean)
                ]),
                self.df
            )

    def _create_hand_columns(self) -> None:
        self.df = self._time_operation("create_hand_nesw_columns", create_hand_nesw_columns, self.df)
        self.df = self._time_operation("create_suit_nesw_columns", create_suit_nesw_columns, self.df)
        self.df = self._time_operation("create_hands_lists_column", create_hands_lists_column, self.df)

    def perform_deal_augmentations(self) -> pl.DataFrame:
        """Main method to perform all deal augmentations"""
        t_start = time.time()
        print(f"Starting deal augmentations")
        
        self._add_default_columns()
        self._create_dealer()
        self._create_vulnerability()
        self._create_hand_columns()
        
        print(f"Deal augmentations complete: {time.time() - t_start:.2f} seconds")
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
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_cards(self) -> None:
        if 'C_NSA' not in self.df.columns:
            self.df = self._time_operation("create C_NSA", DealToCards, self.df)

    def _create_hcp(self) -> None:
        if 'HCP_N_C' not in self.df.columns:
            self.df = self._time_operation("create HCP", CardsToHCP, self.df)

    def _create_quick_tricks(self) -> None:
        if 'QT_N_C' not in self.df.columns:
            self.df = self._time_operation("create QT", CardsToQuickTricks, self.df)

    def _create_suit_lengths(self) -> None:
        if 'SL_N_C' not in self.df.columns:
            sl_nesw_columns = [
                pl.col(f"Suit_{direction}_{suit}").str.len_chars().alias(f"SL_{direction}_{suit}")
                for direction in "NESW"
                for suit in "SHDC"
            ]
            self.df = self._time_operation(
                "create SL_[NESW]_[SHDC]",
                lambda df: df.with_columns(sl_nesw_columns),
                self.df
            )

    def _create_pair_suit_lengths(self) -> None:
        if 'SL_NS_C' not in self.df.columns:
            sl_ns_ew_columns = [
                pl.sum_horizontal(f"SL_{pair[0]}_{suit}", f"SL_{pair[1]}_{suit}").alias(f"SL_{pair}_{suit}")
                for pair in ['NS', 'EW']
                for suit in "SHDC"
            ]
            self.df = self._time_operation(
                "create SL_(NS|EW)_[SHDC]",
                lambda df: df.with_columns(sl_ns_ew_columns),
                self.df
            )

    def _create_suit_length_arrays(self) -> None:
        if 'SL_N_CDHS' not in self.df.columns:
            def create_sl_arrays(df: pl.DataFrame, direction: str) -> dict:
                cdhs_l = df[[f"SL_{direction}_{s}" for s in 'CDHS']].rows()
                ml_li_l = [sorted([(l,i) for i,l in enumerate(r)], reverse=True) for r in cdhs_l]
                ml_l = [[t2[0] for t2 in t4] for t4 in ml_li_l]
                ml_i_l = [[t2[1] for t2 in t4] for t4 in ml_li_l]
                return {
                    f'SL_{direction}_CDHS': (cdhs_l, pl.Array(pl.UInt8, shape=(4,))),
                    f'SL_{direction}_CDHS_SJ': (['-'.join(map(str,r)) for r in cdhs_l], pl.String),
                    f"SL_{direction}_ML": (ml_l, pl.Array(pl.UInt8, shape=(4,))),
                    f"SL_{direction}_ML_SJ": (['-'.join(map(str,r)) for r in ml_l], pl.String),
                    f"SL_{direction}_ML_I": (ml_i_l, pl.Array(pl.UInt8, shape=(4,))),
                    f"SL_{direction}_ML_I_SJ": (['-'.join(map(str,r)) for r in ml_i_l], pl.String)
                }

            for d in 'NESW':
                arrays = create_sl_arrays(self.df, d)
                self.df = self._time_operation(
                    f"create SL_{d} arrays",
                    lambda df: df.with_columns([
                        pl.Series(name, data, dtype) for name, (data, dtype) in arrays.items()
                    ]),
                    self.df
                )

    def _create_distribution_points(self) -> None:
        if 'DP_N_C' not in self.df.columns:
            # Calculate individual suit DPs
            dp_columns = [
                pl.when(pl.col(f"SL_{direction}_{suit}") == 0).then(3)
                .when(pl.col(f"SL_{direction}_{suit}") == 1).then(2)
                .when(pl.col(f"SL_{direction}_{suit}") == 2).then(1)
                .otherwise(0)
                .alias(f"DP_{direction}_{suit}")
                for direction in "NESW"
                for suit in "SHDC"
            ]
            self.df = self._time_operation(
                "create DP columns",
                lambda df: df.with_columns(dp_columns)
                .with_columns([
                    (pl.col(f'DP_{d}_S')+pl.col(f'DP_{d}_H')+pl.col(f'DP_{d}_D')+pl.col(f'DP_{d}_C')).alias(f'DP_{d}')
                    for d in "NESW"
                ])
                .with_columns([
                    (pl.col('DP_N')+pl.col('DP_S')).alias('DP_NS'),
                    (pl.col('DP_E')+pl.col('DP_W')).alias('DP_EW'),
                ])
                .with_columns([
                    (pl.col(f'DP_N_{s}') + pl.col(f'DP_S_{s}')).alias(f'DP_NS_{s}')
                    for s in 'SHDC'
                ] + [
                    (pl.col(f'DP_E_{s}') + pl.col(f'DP_W_{s}')).alias(f'DP_EW_{s}')
                    for s in 'SHDC'
                ]),
                self.df
            )

    def _create_total_points(self) -> None:
        if 'Total_Points_N_C' not in self.df.columns:
            print("Todo: Don't forget to adjust Total_Points for singleton king and doubleton queen.")
            self.df = self._time_operation(
                "create Total_Points",
                lambda df: df.with_columns([
                    (pl.col(f'HCP_{d}_{s}')+pl.col(f'DP_{d}_{s}')).alias(f'Total_Points_{d}_{s}')
                    for d in 'NESW'
                    for s in 'SHDC'
                ])
                .with_columns([
                    (pl.sum_horizontal([f'Total_Points_{d}_{s}' for s in 'SHDC'])).alias(f'Total_Points_{d}')
                    for d in 'NESW'
                ])
                .with_columns([
                    (pl.col('Total_Points_N')+pl.col('Total_Points_S')).alias('Total_Points_NS'),
                    (pl.col('Total_Points_E')+pl.col('Total_Points_W')).alias('Total_Points_EW'),
                ]),
                self.df
            )

    def _create_max_suit_lengths(self) -> None:
        if 'SL_Max_NS' not in self.df.columns:
            sl_cols = [('_'.join(['SL_Max',d]), ['_'.join(['SL',d,s]) for s in SHDC]) 
                      for d in NS_EW]
            for d in sl_cols:
                self.df = self._time_operation(
                    f"create {d[0]}",
                    lambda df: df.with_columns(
                        pl.Series(d[0], [d[1][l.index(max(l))] for l in df[d[1]].rows()])
                    ),
                    self.df
                )

    def _create_quality_indicators(self) -> None:
        series_expressions = [
            pl.Series(
                f"{series_type}_{direction}_{suit}",
                criteria(
                    self.df[f"SL_{direction}_{suit}"],
                    self.df[f"HCP_{direction}_{suit}"]
                ),
                pl.Boolean
            )
            for direction in "NESW"
            for suit in "SHDC"
            for series_type, criteria in {**self.suit_quality_criteria, **self.stopper_criteria}.items()
        ]

        self.df = self._time_operation(
            "create quality indicators",
            lambda df: df.with_columns(series_expressions)
            .with_columns([
                pl.lit(False).alias(f"Forcing_One_Round"),
                pl.lit(False).alias(f"Opponents_Cannot_Play_Undoubled_Below_2N"),
                pl.lit(False).alias(f"Forcing_To_2N"),
                pl.lit(False).alias(f"Forcing_To_3N"),
            ]),
            self.df
        )

    def _create_balanced_indicators(self) -> None:
        self.df = self._time_operation(
            "create balanced indicators",
            lambda df: df.with_columns([
                pl.Series(
                    f"Balanced_{direction}",
                    df[f"SL_{direction}_ML_SJ"].is_in(['4-3-3-3','4-4-3-2']) |
                    (df[f"SL_{direction}_ML_SJ"].is_in(['5-3-3-2','5-4-2-2']) & 
                     (df[f"SL_{direction}_C"].eq(5) | df[f"SL_{direction}_D"].eq(5))),
                    pl.Boolean
                )
                for direction in 'NESW'
            ]),
            self.df
        )

    def perform_hand_augmentations(self) -> pl.DataFrame:
        """Main method to perform all hand augmentations"""
        t_start = time.time()
        print(f"Starting hand augmentations")
        
        self._create_cards()
        self._create_hcp()
        self._create_quick_tricks()
        self._create_suit_lengths()
        self._create_pair_suit_lengths()
        self._create_suit_length_arrays()
        self._create_distribution_points()
        self._create_total_points()
        self._create_max_suit_lengths()
        self._create_quality_indicators()
        self._create_balanced_indicators()
        
        print(f"Hand augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class DD_SD_Augmenter:
    def __init__(self, df: pl.DataFrame, hrs_cache_df: pl.DataFrame, sd_productions: int = 40, max_adds: Optional[int] = None, output_progress: Optional[bool] = True, progress: Optional[Any] = None, lock_func: Optional[Callable[..., pl.DataFrame]] = None):
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_adds = max_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _process_scores_and_tricks(self) -> pl.DataFrame:
        all_scores_d, scores_d, scores_df = self._time_operation("calculate_scores", calculate_scores)
        dd_par_df = self._time_operation(
            "calculate_ddtricks_par_scores", 
            calculate_ddtricks_par_scores, 
            self.df, self.hrs_cache_df, self.max_adds, self.output_progress, self.progress
        )

        if not dd_par_df.is_empty():
            self.hrs_cache_df = update_hrs_cache_df(self.hrs_cache_df, dd_par_df)

        sd_dfs_d, sd_df = self._time_operation(
            "calculate_sd_probs",
            calculate_sd_probs,
            self.df, self.hrs_cache_df, self.sd_productions, self.max_adds, self.progress
        )

        if not sd_df.is_empty():
            self.hrs_cache_df = update_hrs_cache_df(self.hrs_cache_df, sd_df)

        self.df = self.df.join(self.hrs_cache_df, on=['PBN','Dealer','Vul'], how='inner') # on='PBN', how='left' or on=['PBN','Dealer','Vul'], how='inner'

        # create DD_(NS|EW)_[SHDCN] which is the max of NS or EW for each strain
        self.df = self.df.with_columns(
            pl.max_horizontal(f"DD_{pair[0]}_{strain}",f"DD_{pair[1]}_{strain}").alias(f"DD_{pair}_{strain}")
            for pair in ['NS','EW']
            for strain in "SHDCN"
        )

        self.df = self._time_operation(
            "calculate_sd_expected_values",
            calculate_sd_expected_values,
            self.df, scores_df
        )

        best_contracts_df = create_best_contracts(self.df)
        assert self.df.height == best_contracts_df.height, f"{self.df.height} != {best_contracts_df.height}"
        self.df = pl.concat([self.df, best_contracts_df], how='horizontal')
        del best_contracts_df        

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
        print(f"Starting DD/SD trick augmentations")
        
        self.df, self.hrs_cache_df = self._process_scores_and_tricks()
        
        print(f"DD/SD trick augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df, self.hrs_cache_df


class AllContractsAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_ct_types(self) -> None:
        if 'CT_N_C' not in self.df.columns:
            ct_columns = [
                pl.when(pl.col(f"DD_{direction}_{strain}") < 7).then(pl.lit("Pass"))
                .when((pl.col(f"DD_{direction}_{strain}") == 11) & (strain in ['C', 'D'])).then(pl.lit("Game"))
                .when((pl.col(f"DD_{direction}_{strain}").is_in([10,11])) & (strain in ['H', 'S'])).then(pl.lit("Game"))
                .when((pl.col(f"DD_{direction}_{strain}").is_in([9,10,11])) & (strain == 'N')).then(pl.lit("Game"))
                .when(pl.col(f"DD_{direction}_{strain}") == 12).then(pl.lit("SSlam"))
                .when(pl.col(f"DD_{direction}_{strain}") == 13).then(pl.lit("GSlam"))
                .otherwise(pl.lit("Partial"))
                .alias(f"CT_{direction}_{strain}")
                for direction in "NESW"
                for strain in "SHDCN"
            ]
            self.df = self._time_operation(
                "create CT columns",
                lambda df: df.with_columns(ct_columns),
                self.df
            )

    # create CT boolean columns from CT columns
    def _create_ct_booleans(self) -> None:
        if 'CT_N_C_Game' not in self.df.columns:
            ct_boolean_columns = [
                pl.col(f"CT_{direction}_{strain}").eq(pl.lit(contract_type))
                .alias(f"CT_{direction}_{strain}_{contract_type}")
                for direction in "NESW"
                for strain in "SHDCN"
                for contract_type in ["Pass","Game","SSlam","GSlam","Partial"]
            ]
            self.df = self._time_operation(
                "create CT boolean columns",
                lambda df: df.with_columns(ct_boolean_columns),
                self.df
            )
            
        # Create CT boolean columns for pair directions (NS and EW)
        if 'CT_NS_C_Game' not in self.df.columns:
            ct_pair_boolean_columns = [
                (pl.col(f"CT_{pair_direction[0]}_{strain}_{contract_type}") | 
                 pl.col(f"CT_{pair_direction[1]}_{strain}_{contract_type}"))
                .alias(f"CT_{pair_direction}_{strain}_{contract_type}")
                for pair_direction in ["NS", "EW"]
                for strain in "SHDCN"
                for contract_type in ["Pass","Game","SSlam","GSlam","Partial"]
            ]
            self.df = self._time_operation(
                "create CT pair boolean columns",
                lambda df: df.with_columns(ct_pair_boolean_columns),
                self.df
            )

    def perform_all_contracts_augmentations(self) -> pl.DataFrame:
        """Main method to perform AllContracts augmentations"""
        t_start = time.time()
        print(f"Starting AllContracts augmentations")

        self._create_ct_types()
        self._create_ct_booleans()

        print(f"AllContracts augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class FinalContractAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.declarer_to_LHO_d = {None:None,'N':'E','E':'S','S':'W','W':'N'}
        self.declarer_to_dummy_d = {None:None,'N':'S','E':'W','S':'N','W':'E'}
        self.declarer_to_RHO_d = {None:None,'N':'W','E':'N','S':'E','W':'S'}
        self.vul_conditions = {
            'NS': pl.col('Vul').is_in(['N_S', 'Both']),
            'EW': pl.col('Vul').is_in(['E_W', 'Both'])
        }

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _process_contract_columns(self) -> None:
        self.df = self._time_operation(
            "convert_contract_to_contract",
            lambda df: df.with_columns(
                pl.Series('Contract', convert_contract_to_contract(df), pl.String, strict=False)
            ),
            self.df
        )

        # todo: move this section to contract established class
        self.df = self._time_operation(
            "convert_contract_parts",
            lambda df: df.with_columns([
                pl.Series('Declarer_Direction', convert_contract_to_declarer(df), pl.String, strict=False),
                pl.Series('Declarer_Pair_Direction', convert_contract_to_pair_declarer(df), pl.String, strict=False),
                pl.Series('Vul_Declarer', convert_contract_to_vul_declarer(df), pl.Boolean, strict=False),
                pl.Series('BidLvl', convert_contract_to_level(df), pl.UInt8, strict=False),
                pl.Series('BidSuit', convert_contract_to_strain(df), pl.String, strict=False),
                pl.Series('Dbl', convert_contract_to_dbl(df), pl.String, strict=False),
            ]),
            self.df
        )

        # todo: move this to table established class? to_ml() func?
        assert 'LHO_Direction' not in self.df.columns
        assert 'Dummy_Direction' not in self.df.columns
        assert 'RHO_Direction' not in self.df.columns
        print(self.df['Declarer_Direction'].value_counts())
        self.df = self._time_operation(
            "convert_declarer_to_directions",
            lambda df: df.with_columns([
                pl.col('Declarer_Direction').replace_strict(self.declarer_to_LHO_d).alias('LHO_Direction'),
                pl.col('Declarer_Direction').replace_strict(self.declarer_to_dummy_d).alias('Dummy_Direction'),
                pl.col('Declarer_Direction').replace_strict(self.declarer_to_RHO_d).alias('RHO_Direction'),
            ]),
                self.df
            )

        # todo: move this to table established class? to_ml() func?
        assert 'N' not in self.df.columns
        assert 'E' not in self.df.columns
        assert 'S' not in self.df.columns
        assert 'W' not in self.df.columns
        self.df = self._time_operation(
                "rename_players",
                lambda df: df.rename({'Player_Name_N':'N','Player_Name_E':'E','Player_Name_S':'S','Player_Name_W':'W'}),
                self.df
            )

    # create ContractType column using final contract
    def _create_contract_types(self) -> None:
        self.df = self._time_operation(
            "create_contract_types",
            lambda df: df.with_columns(
                pl.when(pl.col('Contract').eq('PASS')).then(pl.lit("Pass"))
                .when(pl.col('BidLvl').eq(5) & pl.col('BidSuit').is_in(['C', 'D'])).then(pl.lit("Game"))
                .when(pl.col('BidLvl').is_in([4,5]) & pl.col('BidSuit').is_in(['H', 'S'])).then(pl.lit("Game"))
                .when(pl.col('BidLvl').is_in([3,4,5]) & pl.col('BidSuit').eq('N')).then(pl.lit("Game"))
                .when(pl.col('BidLvl').eq(6)).then(pl.lit("SSlam"))
                .when(pl.col('BidLvl').eq(7)).then(pl.lit("GSlam"))
                .otherwise(pl.lit("Partial"))
                .alias('ContractType')
            ),
            self.df
        )

    # todo: move this to contract established class
    def _create_declarer_columns(self) -> None:
        self.df = self._time_operation(
            "convert_declarer_columns",
            lambda df: df.with_columns([
                pl.Series('Declarer_Name', convert_declarer_to_DeclarerName(df), pl.String, strict=False),
                pl.Series('Declarer_ID', convert_declarer_to_DeclarerID(df), pl.String, strict=False),
            ]),
            self.df
        )

    # todo: move this to contract established class
    def _create_result_columns(self) -> None:
        if 'Result' not in self.df.columns:
            if 'Tricks' in self.df.columns:
                self.df = self._time_operation(
                    "convert_tricks_to_result",
                    lambda df: df.with_columns(
                        pl.Series('Result', convert_tricks_to_result(df), pl.Int8, strict=False)
                    ),
                    self.df
                )
            else:
                assert 'Contract' in self.df.columns, 'Contract column is required to create Result column.'
                # todo: create assert that result is in Contract.
                self.df = self._time_operation(
                    "convert_contract_to_result",
                    lambda df: df.with_columns(
                        pl.Series('Result', convert_contract_to_result(df), pl.Int8, strict=False)
                    ),
                    self.df
                )

        if 'Tricks' not in self.df.columns:
            self.df = self._time_operation(
                "convert_contract_to_tricks",
                lambda df: df.with_columns(
                    pl.Series('Tricks', convert_contract_to_tricks(df), pl.UInt8, strict=False)
                ),
                self.df
            )

    # todo: move this to contract established class
    def _create_dd_columns(self) -> None:
        if 'DD_Tricks' not in self.df.columns:
            self.df = self._time_operation(
                "convert_contract_to_DD_Tricks",
                lambda df: df.with_columns([
                    pl.Series('DD_Tricks', convert_contract_to_DD_Tricks(df), pl.UInt8, strict=False),
                    pl.Series('DD_Tricks_Dummy', convert_contract_to_DD_Tricks_Dummy(df), pl.UInt8, strict=False),
                ]),
                self.df
            )

        if 'DD_Score_NS' not in self.df.columns:
            self.df = self._time_operation(
                "convert_contract_to_DD_Score_Ref",
                convert_contract_to_DD_Score_Ref,
                self.df
            )
    # todo: move this to contract established class
    def _create_ev_columns(self) -> None:
        max_expressions = []
        for pd in ['NS', 'EW']:
            max_expressions.extend(self._create_ev_expressions_for_pair(pd))

        self.df = self.df.with_columns(max_expressions)

        ev_columns = f'^EV_Max_(NS|EW)$'
        max_expr, col_expr = max_horizontal_and_col(self.df, ev_columns)
        self.df = self.df.with_columns([
            max_expr.alias('EV_Max'),
            col_expr.alias('EV_Max_Col'),
        ])
        
        self.df = self._time_operation(
            "create_ev_columns",
            lambda df: df.with_columns([
                pl.when(pl.col('Declarer_Pair_Direction').eq('NS')).then(pl.col('EV_Max_NS')).otherwise(pl.col('EV_Max_EW')).alias('EV_Max_Declarer'),
                pl.when(pl.col('Declarer_Pair_Direction').eq('NS')).then(pl.col('EV_Max_Col_NS')).otherwise(pl.col('EV_Max_Col_EW')).alias('EV_Max_Col_Declarer'),
            ]),
            self.df
        )


    # todo: move this to contract established class
    def _create_ev_expressions_for_pair(self, pd: str) -> List:
        expressions = []
        expressions.extend(self._create_basic_ev_expressions(pd))
        
        for dd in pd:
            expressions.extend(self._create_declarer_ev_expressions(pd, dd))
            
            for s in 'SHDCN':
                expressions.extend(self._create_strain_ev_expressions(pd, dd, s))
                
                for l in range(1, 8):
                    expressions.extend(self._create_level_ev_expressions(pd, dd, s, l))
        
        return expressions

    # todo: move this to contract established class
    def _create_basic_ev_expressions(self, pd: str) -> List:
        return [
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_V_Max'))
              .otherwise(pl.col(f'EV_{pd}_NV_Max'))
              .alias(f'EV_Max_{pd}'),
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_V_Max_Col'))
              .otherwise(pl.col(f'EV_{pd}_NV_Max_Col'))
              .alias(f'EV_Max_Col_{pd}')
        ]

    # todo: move this to contract established class
    def _create_declarer_ev_expressions(self, pd: str, dd: str) -> List:
        return [
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_{dd}_V_Max'))
              .otherwise(pl.col(f'EV_{pd}_{dd}_NV_Max'))
              .alias(f'EV_{pd}_{dd}_Max'),
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_{dd}_V_Max_Col'))
              .otherwise(pl.col(f'EV_{pd}_{dd}_NV_Max_Col'))
              .alias(f'EV_{pd}_{dd}_Max_Col')
        ]

    # todo: move this to contract established class
    def _create_strain_ev_expressions(self, pd: str, dd: str, s: str) -> List:
        return [
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_{dd}_{s}_V_Max'))
              .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_NV_Max'))
              .alias(f'EV_{pd}_{dd}_{s}_Max'),
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_{dd}_{s}_V_Max_Col'))
              .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_NV_Max_Col'))
              .alias(f'EV_{pd}_{dd}_{s}_Max_Col')
        ]

    # todo: move this to contract established class
    def _create_level_ev_expressions(self, pd: str, dd: str, s: str, l: int) -> List:
        return [
            pl.when(self.vul_conditions[pd])
            .then(pl.col(f'EV_{pd}_{dd}_{s}_{l}_V'))
            .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_{l}_NV'))
            .alias(f'EV_{pd}_{dd}_{s}_{l}')
        ]

    def _create_score_columns(self) -> None:

        if 'Score' not in self.df.columns:
            if 'Score_NS' in self.df.columns:
                assert 'Score_EW' in self.df.columns, "Score_EW does not exist but Score_NS does."
                self.df = self._time_operation(
                    "convert_score_nsew_to_score",
                    lambda df: df.with_columns([
                        pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
                        .then(pl.col('Score_NS'))
                        .otherwise(pl.col('Score_EW')) # assuming Score_EW can be a score, 0 (PASS) or None?
                        .alias('Score'),
                    ]),
                    self.df
                )
            else:
                # neither 'Score' nor 'Score_NS' exist.
                all_scores_d, scores_d, scores_df = calculate_scores()
                self.df = self._time_operation(
                    "convert_contract_to_score",
                    lambda df: df.with_columns([
                        pl.struct(['BidLvl', 'BidSuit', 'Tricks', 'Vul_Declarer', 'Dbl'])
                            .map_elements(lambda x: all_scores_d.get(tuple(x.values()),0), # default becomes 0. ok? should only occur in the case of null (PASS).
                                        return_dtype=pl.Int16)
                            .alias('Score'),
                    ]),
                    self.df
                )

        if 'Score_NS' not in self.df.columns:
            self.df = self._time_operation(
                "convert_score_to_score",
                # lambda df: df.with_columns([
                #     pl.col('Score').alias('Score_NS'),
                #     pl.col('Score').neg().alias('Score_EW')
                # ]),
                lambda df: df.with_columns([
                    pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
                    .then(pl.col('Score'))
                    .otherwise(-pl.col('Score'))
                    .alias('Score_NS'),
                    pl.when(pl.col('Declarer_Pair_Direction').eq('EW'))
                    .then(pl.col('Score'))
                    .otherwise(-pl.col('Score'))
                    .alias('Score_EW')
                ]),
                self.df
            )

    def _create_score_diff_columns(self) -> None:
        # First create the initial diff columns
        self.df = self._time_operation(
            "create_initial_diff_columns",
            lambda df: df.with_columns([
                pl.Series('Par_Diff_NS', (df['Score_NS']-df['Par_NS']), pl.Int16),
                pl.Series('Par_Diff_EW', (df['Score_EW']-df['Par_EW']), pl.Int16),
                pl.Series('DD_Tricks_Diff', (df['Tricks'].cast(pl.Int8)-df['DD_Tricks'].cast(pl.Int8)), pl.Int8, strict=False),
                pl.Series('EV_Max_Diff_NS', df['Score_NS'] - df['EV_Max_NS'], pl.Float32),
                pl.Series('EV_Max_Diff_EW', df['Score_EW'] - df['EV_Max_EW'], pl.Float32),
            ]),
            self.df
        )

        # Then create Par_Diff_EW using the now-existing Par_Diff_NS
        self.df = self._time_operation(
            "create_parscore_diff_ew",
            lambda df: df.with_columns([
                pl.Series('Par_Diff_EW', -df['Par_Diff_NS'], pl.Int16)
            ]),
            self.df
        )

    # todo: would be interesting to enhance this for any contracts and then move into all contract class
    def _create_lott(self) -> None:
        if 'LoTT' not in self.df.columns:
            self.df = self._time_operation("create LoTT", calculate_LoTT, self.df)

    def _perform_legacy_renames(self) -> None:
        self.df = self._time_operation(
            "perform legacy renames",
            Perform_Legacy_Renames,
            self.df
        )


    def _create_position_columns(self) -> None:
        # these augmentations should not already exist.
        assert 'Direction_OnLead' not in self.df.columns
        assert 'Opponent_Pair_Direction' not in self.df.columns
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
            lambda df: df.with_columns([
                pl.struct(['Declarer_Direction', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Declarer_Direction'] is None else r[f'Player_ID_{r["Declarer_Direction"]}'],
                    return_dtype=pl.String
                ).alias('Declarer'),
            ])
            .with_columns([
                pl.col('Declarer_Direction').replace_strict(NextPosition).alias('Direction_OnLead'),
            ])
            .with_columns([
                pl.concat_str([
                    pl.lit('EV'),
                    pl.col('Declarer_Pair_Direction'),
                    pl.col('Declarer_Direction'),
                    pl.col('BidSuit'),
                    pl.col('BidLvl').cast(pl.String),
                ], separator='_').alias('EV_Score_Col_Declarer'),
                
                # pl.when(pl.col('Declarer_Pair_Direction').eq(pl.lit('NS')))
                # .then(pl.col('Score_NS'))
                # .otherwise(pl.col('Score_EW'))
                # .alias('Score_Declarer'),

                pl.struct(['Contract','Declarer_Pair_Direction', 'Score_NS', 'Score_EW']).map_elements(
                    lambda r: 0 if r['Contract'] == 'PASS' else None if r['Declarer_Pair_Direction'] is None else r[f'Score_{r["Declarer_Pair_Direction"]}'],
                    return_dtype=pl.Int16
                ).alias('Score_Declarer'),
                
                pl.when(pl.col('Declarer_Pair_Direction').eq(pl.lit('NS')))
                .then(pl.col('Par_NS'))
                .otherwise(pl.col('Par_EW'))
                .alias('Par_Declarer'),
               
                *[
                    # Note: this is how to create a column name to dynamically choose a value from multiple columns on a row-by-row basis.
                    # For each t (0 ... 13), build a new column which looks up the value
                    # from the column whose name is dynamically built from the row values.
                    # If BidSuit is None, return None.
                    pl.struct([
                        pl.col("Declarer_Pair_Direction"),
                        pl.col("Declarer_Direction"),
                        pl.col("BidSuit"),
                        pl.col("^Probs_.*$")
                    ]).map_elements(
                        # crazy, crazy. current_t is needed because map_elements is a lambda and not a function. otherwise t is always 13!
                        lambda row, current_t=t: None if row["BidSuit"] is None 
                                                  else row[f'Probs_{row["Declarer_Pair_Direction"]}_{row["Declarer_Direction"]}_{row["BidSuit"]}_{current_t}'],
                        return_dtype=pl.Float32
                    ).alias(f'Prob_Taking_{t}') # todo: short form of 'Declarer_SD_Probs_Taking_{t}'
                    for t in range(14)
                ]
            ])
            .with_columns([
                pl.col('Declarer_Pair_Direction').replace_strict(PairDirectionToOpponentPairDirection).alias('Opponent_Pair_Direction'),
                pl.col('Direction_OnLead').replace_strict(NextPosition).alias('Direction_Dummy'),
                pl.struct(['Direction_OnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_OnLead'] is None else r[f'Player_ID_{r["Direction_OnLead"]}'],
                    return_dtype=pl.String
                ).alias('OnLead'),
            ])
            .with_columns([
                pl.col('Direction_Dummy').replace_strict(NextPosition).alias('Direction_NotOnLead'),
                pl.struct(['Direction_Dummy', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_Dummy'] is None else r[f'Player_ID_{r["Direction_Dummy"]}'],
                    return_dtype=pl.String
                ).alias('Dummy'),
                pl.col('Score_Declarer').le(pl.col('Par_Declarer')).alias('Defender_Par_GE')
            ])
            .with_columns([
                pl.struct(['Direction_NotOnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_NotOnLead'] is None else r[f"Player_ID_{r["Direction_NotOnLead"]}"],
                    return_dtype=pl.String
                ).alias('NotOnLead')
            ]),
            self.df
        )

    def _create_board_result_columns(self) -> None:
        print(self.df.filter(pl.col('Result').is_null() | pl.col('Tricks').is_null())
              ['Contract','Declarer_Direction','Vul_Declarer','iVul','Score_NS','BidLvl','Result','Tricks'])
        
        all_scores_d, scores_d, scores_df = calculate_scores() # todo: put this in __init__?
        
        self.df = self._time_operation(
            "create board result columns",
            lambda df: df.with_columns([
                pl.struct(['EV_Score_Col_Declarer','^EV_(NS|EW)_[NESW]_[SHDCN]_[1-7]$'])
                    .map_elements(lambda x: None if x['EV_Score_Col_Declarer'] is None else x[x['EV_Score_Col_Declarer']],
                                return_dtype=pl.Float32).alias('EV_Score_Declarer'),
                pl.struct(['BidLvl', 'BidSuit', 'Tricks', 'Vul_Declarer', 'Dbl'])
                    .map_elements(lambda x: all_scores_d.get(tuple(x.values()),0), # default becomes 0. ok? should only occur in the case of null (PASS).
                                return_dtype=pl.Int16)
                    .alias('Computed_Score_Declarer'),

                pl.struct(['Contract', 'Result', 'Score_NS', 'BidLvl', 'BidSuit', 'Dbl','Declarer_Direction', 'Vul_Declarer']).map_elements(
                    lambda r: None if r['Contract'] is None else 0 if r['Contract'] == 'PASS' else r['Score_NS'] if r['Result'] is None else score(
                        r['BidLvl'] - 1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), 'NESW'.index(r['Declarer_Direction']),
                        r['Vul_Declarer'], r['Result'], True),return_dtype=pl.Int16).alias('Computed_Score_Declarer2'),
            ]),
            self.df
        )
        # todo: can remove df['Computed_Score_Declarer2'] after assert has proven equality
        # if asserts, may be due to Result or Tricks having nulls.
        assert self.df['Computed_Score_Declarer'].eq(self.df['Computed_Score_Declarer2']).all()

    def _create_trick_columns(self) -> None:
        self.df = self._time_operation(
            "create trick columns",
            lambda df: df.with_columns([
                (pl.col('Result') > 0).alias('OverTricks'),
                (pl.col('Result') == 0).alias('JustMade'),
                (pl.col('Result') < 0).alias('UnderTricks'),
                pl.col('Tricks').alias('Tricks_Declarer'),
                (pl.col('Tricks') - pl.col('DD_Tricks')).alias('Tricks_DD_Diff_Declarer'),
            ]),
            self.df
        )

    def _create_rating_columns(self) -> None:
        self.df = self._time_operation(
            "create rating columns",
            lambda df: df.with_columns([
                pl.col('Tricks_DD_Diff_Declarer')
                    .mean()
                    .over('Number_Declarer')
                    .alias('Declarer_Rating'),

                pl.col('Defender_Par_GE')
                    .cast(pl.Float32)
                    .mean()
                    .over('OnLead')
                    .alias('Defender_OnLead_Rating'),

                pl.col('Defender_Par_GE')
                    .cast(pl.Float32)
                    .mean()
                    .over('NotOnLead')
                    .alias('Defender_NotOnLead_Rating')
            ]),
            self.df
        )

    def perform_final_contract_augmentations(self) -> pl.DataFrame:
        """Main method to perform final contract augmentations"""
        t_start = time.time()
        print(f"Starting final contract augmentations")

        self._process_contract_columns()
        self._create_contract_types()
        self._create_declarer_columns()
        self._create_result_columns()
        self._create_score_columns()
        self._create_dd_columns()
        self._create_ev_columns()
        self._create_score_diff_columns()
        self._create_lott() # todo: would be interesting to create lott for all contracts and then move into AllContractsAugmenter
        self._perform_legacy_renames()
        self._create_position_columns()
        self._create_board_result_columns()
        self._create_trick_columns()
        self._create_rating_columns()

        print(f"Final contract augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class MatchPointAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.discrete_score_columns = [] # ['DD_Score_NS', 'EV_Max_NS'] # calculate matchpoints for these columns which change with each row's Score_NS
        self.dd_score_columns = [f'DD_Score_{l}{s}_{d}' for d in 'NESW' for s in 'SHDCN' for l in range(1,8)]
        self.ev_score_columns = [f'EV_{pd}_{d}_{s}_{l}' for pd in ['NS','EW'] for d in pd for s in 'SHDCN' for l in range(1,8)]
        self.all_score_columns = self.discrete_score_columns + self.dd_score_columns + self.ev_score_columns

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_mp_top(self) -> None:
        if 'MP_Top' not in self.df.columns:
            self.df = self._time_operation(
                "create MP_Top",
                lambda df: df.with_columns(
                    pl.col('Score').count().over(['session_id','PBN','Board']).sub(1).alias('MP_Top')
                ),
                self.df
            )

    def _calculate_matchpoints(self) -> None:
        if 'MP_NS' not in self.df.columns:
            self.df = self._time_operation(
                "calculate matchpoints MP_(NS|EW)",
                lambda df: df.with_columns([
                    pl.col('Score_NS').rank(method='average', descending=False).sub(1)
                        .over(['session_id', 'PBN', 'Board']).alias('MP_NS'),
                    pl.col('Score_EW').rank(method='average', descending=False).sub(1)
                        .over(['session_id', 'PBN', 'Board']).alias('MP_EW'),
                ]),
                self.df
            )

    def _calculate_percentages(self) -> None:
        if 'Pct_NS' not in self.df.columns:
            self.df = self._time_operation(
                "calculate matchpoints percentages",
                lambda df: df.with_columns([
                    (pl.col('MP_NS') / pl.col('MP_Top')).alias('Pct_NS'),
                    (pl.col('MP_EW') / pl.col('MP_Top')).alias('Pct_EW')
                ]),
                self.df
            )

    def _create_declarer_pct(self) -> None:
        if 'Declarer_Pct' not in self.df.columns:
            self.df = self._time_operation(
                "create Declarer_Pct",
                lambda df: df.with_columns(
                    pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
                    .then('Pct_NS')
                    .otherwise('Pct_EW')
                    .alias('Declarer_Pct')
                ),
                self.df
            )

    def _calculate_matchpoints_group(self, series_list: list[pl.Series]) -> pl.Series:
        col_values = series_list[0]
        score_ns_values = series_list[1]
        if col_values.is_null().sum() > 0:
            print(f"Warning: Null values in col_values: {col_values.is_null().sum()}")
        #if score_ns_values.is_null().sum() > 0:
        #    print(f"Warning: Null values in score_ns_values: {score_ns_values.is_null().sum()}")
        # todo: is there a more proper way to handle null values in col_values and score_ns_values?
        score_ns_values = score_ns_values.fill_null(0.0) # todo: why do some have nulls? sitout? adjusted score?
        col_values = col_values.fill_null(0.0) # todo: why do some have nulls? sitout? adjusted score?
        return pl.Series([
            sum(1.0 if val > score else 0.5 if val == score else 0.0 
                for score in score_ns_values)
            for val in col_values
        ])

    def _calculate_all_score_matchpoints(self) -> None:
        t = time.time()
        # if 'Expanded_Scores_List' in self.df.columns: # todo: obsolete?
        #     print('Calculate matchpoints for existing Expanded_Scores_List column.')
        #     self.df = calculate_matchpoint_scores_ns(self.df, self.all_score_columns)
        # else:
        print('Calculate matchpoints over session, PBN, and Board.')
        # calc matchpoints on row-by-row basis
        if self.df['Score_NS'].null_count() > 0:
            print(f"Warning: Null values in score_ns_values: {self.df['Score_NS'].is_null().sum()}")
        # compute matchpoints for DD_Score_[1-7][CDHSN]_[NESW] and EV_(NS|EW)_[NESW]_[CDHSN]_[1-7] and misc columns.
        for col in self.all_score_columns + ['DD_Score_NS', 'DD_Score_EW', 'Par_NS', 'Par_EW']:
            assert 'MP_'+col not in self.df.columns, f"Column 'MP_{col}' already exists in DataFrame"
            self.df = self.df.with_columns([
                    pl.map_groups(
                        exprs=[col, 'Score_NS' if '_NS' in col or col[-1] in 'NS' else 'Score_EW'],
                        function=self._calculate_matchpoints_group,
                        return_dtype=pl.Float64,
                    ).over(['session_id', 'PBN', 'Board']).alias(f'MP_{col}')
                ])
        # compute matchpoints for declarer orientation columns
        for col in [('DD_Score_Declarer','Score_Declarer'),('Par_Declarer','Score_Declarer'),('EV_Score_Declarer','Score_Declarer'),('EV_Max_Declarer','Score_Declarer')]:
            assert 'MP_'+col[0] not in self.df.columns, f"Column 'MP_{col[0]}' already exists in DataFrame"
            self.df = self.df.with_columns([
                    pl.map_groups(
                        exprs=col,
                        function=self._calculate_matchpoints_group,
                        return_dtype=pl.Float64,
                    ).over(['session_id', 'PBN', 'Board']).alias(f'MP_{col[0]}')
                ])
        print(f"calculate matchpoints all_score_columns: time:{time.time()-t} seconds")

    def _calculate_mp_pct_from_new_score(self, col: str) -> pl.Series:
        return (pl.col(f'MP_{col}')+pl.col(f'MP_{col}').gt(pl.col(col))+(pl.col(f'MP_{col}').eq(pl.col(col))/2))/(pl.col('MP_Top')+1)

    def _calculate_final_scores(self) -> None:
        t = time.time()
        
        # Calculate DD and Par percentages. Technique is to start calc matchpoints then compare dd/par score with all Score values for the same board and then divide by MP_Top + 1.
        # SELECT Board, Score_NS, DD_Score_NS, MP_DD_Score_NS, DD_Pct_NS, DD_Score_EW, MP_DD_Score_EW, DD_Pct_EW, Par_NS, MP_Par_NS, Par_Pct_NS
        for col in ['DD_Score']:
            for pair in ['NS','EW']:
                col_pair = col + '_' + pair
                # Calculate matchpoints: compare each row's DD_Score with all Score values for the same board.
                board_scores_tuples = self.df.group_by('Board').agg(pl.col(f'Score_{pair}')).rows()
                board_to_scores_d = {board: scores for board, scores in board_scores_tuples}
                self.df = self.df.with_columns(
                    pl.struct([col_pair, 'Board']).map_elements(
                        lambda row: sum(
                            1.0 if row[col_pair] > score else 0.5 if row[col_pair] == score else 0.0
                            for score in board_to_scores_d[row['Board']]
                        ),
                        return_dtype=pl.Float64
                    ).alias(f'MP_{col_pair}')
                )
                self.df = self.df.with_columns([
                    (pl.col(f'MP_{col_pair}')/(pl.col('MP_Top')+1)).alias(f'{col}_Pct_{pair}')
                ])
                assert self.df[f'{col}_Pct_{pair}'].is_between(0,1).all()
        # calculate Par percentages.
        for col in ['Par']:
            for pair in ['NS','EW']:
                col_pair = col + '_' + pair
                self.df = self.df.with_columns([
                    pl.when(pl.col(col_pair) > pl.col(f'Score_{pair}')).then(1.0)
                    .when(pl.col(col_pair) == pl.col(f'Score_{pair}')).then(0.5)
                    .otherwise(0.0)
                    .alias(f"MP_Rank_{col_pair}")
                ])
                self.df = self.df.with_columns([
                    (pl.col(f"MP_Rank_{col_pair}").sum().over('Board')).alias(f'MP_{col_pair}'),
                ])
                self.df = self.df.with_columns([
                    (pl.col(f'MP_{col_pair}')/(pl.col('MP_Top')+1)).alias(f'{col}_Pct_{pair}')
                ])
                assert self.df[f'{col}_Pct_{pair}'].is_between(0,1).all()

        # for declarer orientation scores
        self.df = self.df.with_columns(
            self._calculate_mp_pct_from_new_score('DD_Score_Declarer').alias('MP_DD_Pct_Declarer'),
            self._calculate_mp_pct_from_new_score('Par_Declarer').alias('MP_Par_Pct_Declarer'),
            self._calculate_mp_pct_from_new_score('EV_Score_Declarer').alias('MP_EV_Pct_Declarer'),
            self._calculate_mp_pct_from_new_score('EV_Max_Declarer').alias('MP_EV_Max_Pct_Declarer')
        )
        # todo: assert self.df['MP_DD_Pct_Declarer'].between(0,1).all()
        # todo: assert self.df['MP_Par_Pct_Declarer'].between(0,1).all()
        # todo: assert self.df['MP_EV_Pct_Declarer'].between(0,1).all()
        # todo: assert self.df['MP_EV_Max_Pct_Declarer'].between(0,1).all()

        # Calculate remaining scores and percentages
        operations = [
            #lambda df: df.with_columns((1-pl.col('Par_Pct_NS')).alias('Par_Pct_EW')),
            lambda df: df.with_columns(pl.max_horizontal(f'^MP_DD_Score_[1-7][SHDCN]_[NS]$').alias(f'MP_DD_Score_Max_NS')), # actual max score for NS
            lambda df: df.with_columns(pl.max_horizontal(f'^MP_DD_Score_[1-7][SHDCN]_[EW]$').alias(f'MP_DD_Score_Max_EW')), # actual max score for EW
            lambda df: df.with_columns(pl.max_horizontal(f'^MP_EV_NS_[NS]_[SHDCN]_[1-7]$').alias(f'MP_EV_Max_NS')), # predicted max score for NS
            lambda df: df.with_columns(pl.max_horizontal(f'^MP_EV_EW_[EW]_[SHDCN]_[1-7]$').alias(f'MP_EV_Max_EW')), # predicted max score for EW
            lambda df: df.with_columns([
                (pl.col('MP_DD_Score_Max_NS')/pl.col('MP_Top')).alias('DD_Pct_Max_NS'), # actual max Pct for NS
                (pl.col('MP_DD_Score_Max_EW')/pl.col('MP_Top')).alias('DD_Pct_Max_EW'), # actual max Pct for EW
                #(pl.col('MP_EV_Max_NS')/pl.col('MP_Top')).alias('EV_Pct_Max_NS'),
                #(pl.col('MP_EV_Max_EW')/pl.col('MP_Top')).alias('EV_Pct_Max_EW'),
                # self._calculate_pct_from_new_score('MP_DD_Score_Max_NS').alias('DD_Pct_Max_NS'),
                # self._calculate_pct_from_new_score('MP_DD_Score_Max_EW').alias('DD_Pct_Max_EW'),
                self._calculate_mp_pct_from_new_score('EV_Max_NS').alias('EV_Pct_Max_NS'), # predicted max Pct for NS
                self._calculate_mp_pct_from_new_score('EV_Max_EW').alias('EV_Pct_Max_EW'), # predicted max Pct for EW
            ]),
            # todo: assert self.df['DD_Pct_Max_NS'].between(0,1).all()
            # todo: assert self.df['DD_Pct_Max_EW'].between(0,1).all()
            # todo: assert self.df['EV_Pct_Max_NS'].between(0,1).all()
            # todo: assert self.df['EV_Pct_Max_EW'].between(0,1).all()
            lambda df: df.with_columns([
                (pl.col('Pct_NS')-pl.col('EV_Pct_Max_NS')).alias('EV_Pct_Max_Diff_NS'), # diff between actual Pct and predicted max Pct for NS
                (pl.col('Pct_EW')-pl.col('EV_Pct_Max_EW')).alias('EV_Pct_Max_Diff_EW'), # diff between actual Pct and predicted max Pct for EW
                (pl.col('Pct_NS')-pl.col('DD_Pct_Max_NS')).alias('DD_Pct_Max_Diff_NS'), # diff between actual Pct and max DD Pct for NS
                (pl.col('Pct_EW')-pl.col('DD_Pct_Max_EW')).alias('DD_Pct_Max_Diff_EW'), # diff between actual Pct and max DD Pct for EW
                (pl.col('DD_Pct_Max_NS')-pl.col('EV_Pct_Max_NS')).alias('DD_EV_Pct_Max_Diff_NS'), # diff between max DD Pct and predicted max Pct for NS
                (pl.col('DD_Pct_Max_EW')-pl.col('EV_Pct_Max_EW')).alias('DD_EV_Pct_Max_Diff_EW'), # diff between max DD Pct and predicted max Pct for EW
            ])
        ]

        for operation in operations:
            self.df = operation(self.df)

        print(f"Time to rank expanded scores: {time.time()-t} seconds")

    def perform_matchpoint_augmentations(self) -> pl.DataFrame:
        t_start = time.time()
        print(f"Starting matchpoint augmentations")
        
        self._create_mp_top()
        self._calculate_matchpoints()
        self._calculate_percentages()
        self._create_declarer_pct()
        self._calculate_all_score_matchpoints()
        self._calculate_final_scores()
        
        print(f"Matchpoint augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


# todo: IMP augmentations are not implemented yet
class IMPAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def perform_imp_augmentations(self) -> pl.DataFrame:
        t_start = time.time()
        print(f"Starting IMP augmentations")
        
        print(f"IMP augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class AllHandRecordAugmentations:
    def __init__(self, df: pl.DataFrame, 
                 hrs_cache_df: Optional[pl.DataFrame] = None, 
                 sd_productions: int = 40, 
                 max_adds: Optional[int] = None,
                 output_progress: Optional[bool] = True,
                 progress: Optional[Any] = None,
                 lock_func: Optional[Callable[..., pl.DataFrame]] = None):
        """Initialize the AllAugmentations class with a DataFrame and optional parameters.
        
        Args:
            df: The input DataFrame to augment
            hrs_cache_df: dataframe of cached computes
            sd_productions: Number of single dummy productions to generate
            max_adds: Maximum number of adds to generate
            output_progress: Whether to output progress
            progress: Optional progress indicator object
            lock_func: Optional function for thread safety
        """
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_adds = max_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func

        # instance initialization

        # Double dummy tricks for each player and strain
        dd_cols = {f"DD_{p}_{s}": pl.UInt8 for p in 'NESW' for s in 'CDHSN'}

        # Single dummy probabilities. Note that the order of declarers and suits must match the original schema.
        # The declarer order was found to be N, S, W, E from inspecting the original schema.
        probs_cols = {f"Probs_{pair}_{declarer}_{s}_{i}": pl.Float64 for pair in ['NS', 'EW'] for declarer in 'NESW' for s in 'CDHSN' for i in range(14)}

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
                'Doubled': pl.String, 
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
        print(f"Starting all hand record augmentations on DataFrame with {len(self.df)} rows")
        
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
            self.max_adds, 
            self.output_progress,
            self.progress,
            self.lock_func
        )
        self.df, self.hrs_cache_df = dd_sd_augmenter.perform_dd_sd_augmentations()

        # todo: move this somewhere more sensible.
        self.df = self.df.with_columns(pl.col('ParScore').alias('Par_NS'))
        self.df = self.df.with_columns(pl.col('ParScore').neg().alias('Par_EW'))
        
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

        print(f"All hand records augmentations completed in {time.time() - t_start:.2f} seconds")
        
        return self.df, self.hrs_cache_df


class AllBoardResultsAugmentations:
    def __init__(self, df: pl.DataFrame):
        self.df = df


    def perform_all_board_results_augmentations(self) -> pl.DataFrame:
        """Execute all board results augmentation steps. Input is a fully augmented hand record DataFrame.
        Only relies on columns within brs_df and not any in hrs_df.

        Returns:
            The fully joined and augmented hand record and board results DataFrame.
        """
        t_start = time.time()
        print(f"Starting all board results augmentations on DataFrame with {len(self.df)} rows")
        
        # Step 5: Final contract augmentations
        hand_augmenter = FinalContractAugmenter(self.df)
        self.df = hand_augmenter.perform_final_contract_augmentations()
        
        # Step 6: Matchpoint augmentations
        matchpoint_augmenter = MatchPointAugmenter(self.df)
        self.df = matchpoint_augmenter.perform_matchpoint_augmentations()
        
        # Step 7: IMP augmentations (not implemented yet)
        imp_augmenter = IMPAugmenter(self.df)
        self.df = imp_augmenter.perform_imp_augmentations()
        
        print(f"All board results augmentations completed in {time.time() - t_start:.2f} seconds")
        
        return self.df


class AllAugmentations:
    def __init__(self, df: pl.DataFrame, hrs_cache_df: Optional[pl.DataFrame] = None, sd_productions: int = 40, max_adds: Optional[int] = None, output_progress: Optional[bool] = True, progress: Optional[Any] = None, lock_func: Optional[Callable[..., pl.DataFrame]] = None):
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_adds = max_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func

    def perform_all_augmentations(self) -> pl.DataFrame:
        """Execute all augmentation steps.
        
        Returns:
            The fully joined and augmented hand record and board results DataFrame.
        """
        t_start = time.time()
        print(f"Starting all augmentations on DataFrame with {len(self.df)} rows")

        hand_record_augmenter = AllHandRecordAugmentations(self.df, self.hrs_cache_df, self.sd_productions, self.max_adds, self.output_progress, self.progress, self.lock_func)
        self.df, self.hrs_cache_df = hand_record_augmenter.perform_all_hand_record_augmentations()
        board_results_augmenter = AllBoardResultsAugmentations(self.df)
        self.df = board_results_augmenter.perform_all_board_results_augmentations()

        print(f"All augmentations completed in {time.time() - t_start:.2f} seconds")

        return self.df, self.hrs_cache_df


# def read_parquet_sample(file_path, n_rows=1000, method='head'):
#     """
#     Read a sample of rows from a parquet file.
    
#     Args:
#         file_path: Path to parquet file
#         n_rows: Number of rows to sample
#         method: 'head' for first n rows, 'sample' for random sample, 'tail' for last n rows
    
#     Returns:
#         DataFrame with sampled rows
#     """
#     if method == 'head':
#         return pl.scan_parquet(file_path).limit(n_rows).collect()
#     elif method == 'sample':
#         return pl.scan_parquet(file_path).sample(n_rows).collect()
#     elif method == 'tail':
#         return pl.scan_parquet(file_path).tail(n_rows).collect()
#     else:
#         raise ValueError("method must be 'head', 'sample', or 'tail'")

# def read_parquet_slice(file_path, offset=0, length=1000):
#     """
#     Read a specific slice of rows from a parquet file.
    
#     Args:
#         file_path: Path to parquet file
#         offset: Starting row (0-indexed)
#         length: Number of rows to read
    
#     Returns:
#         DataFrame with sliced rows
#     """
#     return pl.scan_parquet(file_path).slice(offset, length).collect()

# def read_parquet_filtered(file_path, filters=None, n_rows=None):
#     """
#     Read parquet file with filters and optional row limit.
    
#     Args:
#         file_path: Path to parquet file
#         filters: Polars expression for filtering
#         n_rows: Optional limit on number of rows
    
#     Returns:
#         Filtered DataFrame
#     """
#     lazy_df = pl.scan_parquet(file_path)
    
#     if filters is not None:
#         lazy_df = lazy_df.filter(filters)
    
#     if n_rows is not None:
#         lazy_df = lazy_df.limit(n_rows)
    
#     return lazy_df.collect()

# def read_parquet_every_nth(file_path, n=10):
#     """
#     Read every nth row from a parquet file.
    
#     Args:
#         file_path: Path to parquet file
#         n: Take every nth row
    
#     Returns:
#         DataFrame with every nth row
#     """
#     return (pl.scan_parquet(file_path)
#             .with_row_index()
#             .filter(pl.col("index") % n == 0)
#             .drop("index")
#             .collect())

# def read_parquet_by_percentage(file_path, percentage=0.1):
#     """
#     Read a percentage of rows from a parquet file using random sampling.
    
#     Args:
#         file_path: Path to parquet file
#         percentage: Percentage of rows to sample (0.0 to 1.0)
    
#     Returns:
#         DataFrame with sampled rows
#     """
#     return pl.scan_parquet(file_path).sample(fraction=percentage).collect()

# def read_parquet_lazy_info(file_path):
#     """
#     Get information about a parquet file without reading data.
    
#     Args:
#         file_path: Path to parquet file
    
#     Returns:
#         Dictionary with file info
#     """
#     lazy_df = pl.scan_parquet(file_path)
#     schema = lazy_df.collect_schema()
    
#     # Get row count (this does scan the file but doesn't load data)
#     row_count = lazy_df.select(pl.len()).collect().item()
    
#     return {
#         'columns': schema.names(),
#         'dtypes': {name: str(dtype) for name, dtype in schema.items()},
#         'row_count': row_count,
#         'column_count': len(schema)
#     }

# def read_parquet_lazy_select(file_path, columns=None, filters=None, sample_n=None, sample_fraction=None):
#     """
#     Read parquet file using lazy evaluation with column selection and optional operations.
#     This is the PREFERRED approach for reading parquet files.
    
#     Args:
#         file_path: Path to parquet file
#         columns: List of column names to select
#         filters: Polars expression for filtering
#         sample_n: Number of rows to sample
#         sample_fraction: Fraction of rows to sample (0.0 to 1.0)
    
#     Returns:
#         DataFrame with selected columns and applied operations
#     """
#     lazy_df = pl.scan_parquet(file_path)
    
#     # Apply column selection (projection pushdown)
#     if columns is not None:
#         lazy_df = lazy_df.select(columns)
    
#     # Apply filters (predicate pushdown)
#     if filters is not None:
#         lazy_df = lazy_df.filter(filters)
    
#     # Apply sampling
#     if sample_n is not None:
#         lazy_df = lazy_df.sample(n=sample_n)
#     elif sample_fraction is not None:
#         lazy_df = lazy_df.sample(fraction=sample_fraction)
    
#     # Execute the optimized query plan
#     return lazy_df.collect()

# def read_parquet_with_sampling(file_path, columns=None, n_rows=None, method='limit', seed=None):
#     """
#     Read parquet file with sampling, handling LazyFrame compatibility issues.
    
#     Args:
#         file_path: Path to parquet file
#         columns: List of column names to select
#         n_rows: Number of rows to sample/limit
#         method: 'limit' (first n rows), 'collect_sample' (random), 'systematic' (every nth)
#         seed: Random seed for sampling
    
#     Returns:
#         DataFrame with sampled rows
#     """
#     lazy_df = pl.scan_parquet(file_path)
    
#     if columns is not None:
#         lazy_df = lazy_df.select(columns)
    
#     if n_rows is None:
#         return lazy_df.collect()
    
#     if method == 'limit':
#         # Fastest - first n rows
#         return lazy_df.limit(n_rows).collect()
    
#     elif method == 'collect_sample':
#         # True random sampling - requires collecting all data first
#         df = lazy_df.collect()
#         return df.sample(n=n_rows, seed=seed)
    
#     elif method == 'systematic':
#         # Every nth row - memory efficient
#         total_rows = lazy_df.select(pl.len()).collect().item()
#         skip = max(1, total_rows // n_rows)
#         return (
#             lazy_df
#             .with_row_index()
#             .filter(pl.col("index") % skip == 0)
#             .drop("index")
#             .limit(n_rows)
#             .collect()
#         )
    
#     else:
#         raise ValueError("method must be 'limit', 'collect_sample', or 'systematic'")

