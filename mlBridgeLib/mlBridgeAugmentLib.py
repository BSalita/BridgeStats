
# contains functions to augment df with additional columns
# mostly polars functions

import polars as pl
from collections import defaultdict
import sys
import time

import endplay # for __version__
from endplay.parsers import pbn, lin, json
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
from endplay.dds import calc_dd_table, calc_all_tables, par
from endplay.dealer import generate_deals

import mlBridgeLib


def create_hand_nesw_columns(df):    
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


def create_hands_lists_column(df):
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


def create_suit_nesw_columns(df):
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
def OHE_Hands(hands_bin):
    handsbind = defaultdict(list)
    for h in hands_bin:
        for direction,nesw in zip(mlBridgeLib.NESW,h):
            assert nesw[0] is not None and nesw[1] is not None
            handsbind['_'.join(['HB',direction])].append(nesw[0]) # todo: int(nesw[0],2)) # convert binary string to base 2 int
            #for suit,shdc in zip(mlBridgeLib.SHDC,nesw[1]):
            #    assert shdc is not None
            #    handsbind['_'.join(['HCP',direction,suit])].append(shdc)
    return handsbind


# generic function to augment metrics by suits
def Augment_Metric_By_Suits(metrics,metric,dtype=pl.UInt8):
    for d,direction in enumerate(mlBridgeLib.NESW):
        for s,suit in  enumerate(mlBridgeLib.SHDC):
            metrics = metrics.with_columns(
                metrics[metric].map_elements(lambda x: x[1][d][0],return_dtype=dtype).alias('_'.join([metric,direction])),
                metrics[metric].map_elements(lambda x: x[1][d][1][s],return_dtype=dtype).alias('_'.join([metric,direction,suit]))
            )
    for direction in mlBridgeLib.NS_EW:
        metrics = metrics.with_columns((metrics['_'.join([metric,direction[0]])]+metrics['_'.join([metric,direction[1]])]).cast(dtype).alias('_'.join([metric,direction])))
        for s,suit in  enumerate(mlBridgeLib.SHDC):
            metrics = metrics.with_columns((metrics['_'.join([metric,direction[0],suit])]+metrics['_'.join([metric,direction[1],suit])]).cast(dtype).alias('_'.join([metric,direction,suit])))
    #display(metrics.describe())
    return metrics # why is it necessary to return metrics? Isn't it just df?


# calculate dict of contract result scores. each column contains (non-vul,vul) scores for each trick taken. sets are always penalty doubled.
def calculate_scores():

    scores_d = {}
    all_scores_d = {(None,None,None,None,None):0} # PASS

    suit_to_denom = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
    for suit_char in 'SHDCN':
        suit_index = 'CDHSN'.index(suit_char) # [3,2,1,0,4]
        denom = suit_to_denom[suit_index]
        for level in range(1,8): # contract level
            for tricks in range(14):
                result = tricks-6-level
                # sets are always penalty doubled
                scores_d[(level,suit_char,tricks,False)] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(Vul.none)
                scores_d[(level,suit_char,tricks,True)] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(Vul.both)
                # calculate all possible scores
                all_scores_d[(level,suit_char,tricks,False,'')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed,result=result).score(Vul.none)
                all_scores_d[(level,suit_char,tricks,False,'X')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.doubled,result=result).score(Vul.none)
                all_scores_d[(level,suit_char,tricks,False,'XX')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.redoubled,result=result).score(Vul.none)
                all_scores_d[(level,suit_char,tricks,True,'')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed,result=result).score(Vul.both)
                all_scores_d[(level,suit_char,tricks,True,'X')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.doubled,result=result).score(Vul.both)
                all_scores_d[(level,suit_char,tricks,True,'XX')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.redoubled,result=result).score(Vul.both)

    # create score dataframe from dict
    sd = defaultdict(list)
    for suit in 'SHDCN':
        for level in range(1,8):
            for i in range(14):
                sd['_'.join(['Score',str(level)+suit])].append([scores_d[(level,suit,i,False)],scores_d[(level,suit,i,True)]])
    # st.write(all_scores_d)
    scores_df = pl.DataFrame(sd,orient='row')
    # scores_df.index.name = 'Taken'
    return all_scores_d, scores_d, scores_df


def display_double_dummy_deals(deals, dd_result_tables, deal_index=0, max_display=4):
    # Display a few hands and double dummy tables
    for dd, rt in zip(deals[deal_index:deal_index+max_display], dd_result_tables[deal_index:deal_index+max_display]):
        deal_index += 1
        print(f"Deal: {deal_index}")
        print(dd)
        rt.pprint()


# todo: could save a couple seconds by creating dict of deals
def calc_double_dummy_deals(deals, batch_size=40, output_progress=False, progress=None):
    # was the wonkyness due to unique() not having maintain_order=True? Let's see if it behaves now.
    #if isinstance(deals,pl.Series):
    #    deals = deals.to_list() # needed because polars silently ignored the [b:b+batch_size] slicing. WTF?
    all_result_tables = []
    for i,b in enumerate(range(0,len(deals),batch_size)):
        if output_progress:
            if progress:
                percent_complete = int(i*100/len(deals))
                progress.progress(percent_complete,f"{percent_complete}%: Double dummies calculated for {i} of {len(deals)} unique deals.")
            else:
                if i % 1000 == 0:
                    percent_complete = int(i*100/len(deals))
                    print(f"{percent_complete}%: Double dummies calculated for {i} of {len(deals)} unique deals.")
        result_tables = calc_all_tables(deals[b:b+batch_size])
        all_result_tables.extend(result_tables)
    if output_progress: 

        if progress:
            progress.progress(100,f"100%: Double dummies calculated for {len(deals)} unique deals.")
            progress.empty() # hmmm, this removes the progress bar so fast that 100% message won't be seen.
        else:
            print(f"100%: Double dummies calculated for {len(deals)} unique deals.")
    return all_result_tables



def calculate_ddtricks_par_scores(df, hrs_d, scores_d, output_progress=True, progress=None):

    # Calculate double dummy and par
    unique_pbns = df['PBN'].unique(maintain_order=True)
    pbns = [pbn for pbn in unique_pbns if pbn not in hrs_d or 'DD' not in hrs_d[pbn]]
    deals = [Deal(pbn) for pbn in pbns]
    assert all([pbn == dpbn.to_pbn() for pbn,dpbn in zip(pbns,deals)]) # usually a sort order issue which should have been fixed in previous step
    unique_dd_tables = calc_double_dummy_deals(deals, output_progress=output_progress, progress=progress)
    unique_dd_tables_d = {deal.to_pbn():rt for deal,rt in zip(deals,unique_dd_tables)}

    # Create dataframe of par scores using double dummy
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

    par_scores_ns = []
    par_scores_ew = []
    par_contracts = []
    flattened_dd_rows = []
    for pbn, dealer, vul in df[('PBN','Dealer','Vul')].rows():
        if pbn not in hrs_d:
            hrs_d[pbn] = {}
        if 'DD' not in hrs_d[pbn]:
            hrs_d[pbn]['DD'] = unique_dd_tables_d[pbn]
        rt = hrs_d[pbn]['DD']
        # middle arg is board number (if int) otherwise enum vul. Must use Vul.find(v) because some boards have random (vul,dealer).
        parlist = par(rt, VulToEndplayVul_d[vul], DealerToEndPlayDealer_d[dealer])
        # there may be multiple par scores for a given pbn. pbn's may have different (dealer,vul) combinations.
        if 'Par' not in hrs_d[pbn]:
            hrs_d[pbn]['Par'] = {}
        hrs_d[pbn]['Par'][(dealer,vul)] = parlist
        par_scores_ns.append(parlist.score)
        par_scores_ew.append(-parlist.score)
        par_contracts.append([', '.join([str(contract.level) + 'SHDCN'[int(contract.denom)] + contract.declarer.abbr + contract.penalty.abbr + ('' if contract.result == 0 else '+'+str(contract.result) if contract.result > 0 else str(contract.result)) for contract in parlist])])
        # convert endplay's dd table to df by flattening each dd table into rows.
        flattened_row = [item for sublist in zip(*rt.to_list()) for item in sublist]
        flattened_dd_rows.append(flattened_row)
    par_df = pl.DataFrame({'Par_NS': par_scores_ns, 'Par_EW': par_scores_ew, 'ParContract': par_contracts},orient='row')

    # Create column names
    columns = {f'DD_{direction}_{suit}':pl.UInt8 for direction in 'NESW' for suit in 'SHDCN'}

    # Create the DataFrame
    DD_Tricks_df = pl.DataFrame(flattened_dd_rows, schema=columns, orient='row')

    dd_ns_ew_columns = [
        pl.max_horizontal(f"DD_{pair[0]}_{strain}",f"DD_{pair[1]}_{strain}").alias(f"DD_{pair}_{strain}")
        for pair in ['NS','EW']
        for strain in "SHDCN"
    ]
    DD_Tricks_df = DD_Tricks_df.with_columns(dd_ns_ew_columns)

    dd_score_cols = [[scores_d[(level,suit,tricks,vul == 'Both' or (vul != 'None' and direction in vul))] for tricks,vul in zip(DD_Tricks_df['_'.join(['DD',direction,suit])],df['Vul'])] for direction in 'NESW' for suit in 'SHDCN' for level in range(1, 8)]
    dd_score_df = pl.DataFrame(dd_score_cols, schema=['_'.join(['DD_Score', str(l) + s, d]) for d in 'NESW' for s in 'SHDCN' for l in range(1, 8)])
    
    return DD_Tricks_df, par_df, dd_score_df


def constraints(deal):
    return True


def generate_single_dummy_deals(predeal_string, produce, env=dict(), max_attempts=1000000, seed=42, show_progress=True, strict=True, swapping=0):
    
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


def calculate_single_dummy_probabilities(deal, produce=100):

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


# takes 1000 seconds for 100 sd calcs, or 10 sd calcs per second.
def calculate_sd_probs(df, hrs_d, sd_productions=100, progress=None):

    # calculate single dummy probabilities. if already calculated use cache value else update cache with new result.
    sd_dfs_d = {}
    unique_pbns = df['PBN'].unique(maintain_order=True) # todo: unique and not cached: if pbn not in hrs_d or 'SD' not in hrs_d[pbn] then calculate
    #print(unique_df)
    for i,pbn in enumerate(unique_pbns):
        if progress:
            percent_complete = int(i*100/len(unique_pbns))
            progress.progress(percent_complete,f"{percent_complete}%: Single dummies calculated for {i} of {len(unique_pbns)} unique deals using {sd_productions} samples per deal. Takes 1 minute...")
        else:
            if i < 10 or i % 10000 == 0:
                percent_complete = int(i*100/len(unique_pbns))
                print(f"{percent_complete}%: Single dummies calculated for {i} of {len(unique_pbns)} unique deals using {sd_productions} samples per deal.")
        if pbn not in hrs_d:
            hrs_d[pbn] = {}
        if 'SD' not in hrs_d[pbn]:
            #print(pbn)
            if not progress and (i < 10 or i % 10000 == 0):
                t = time.time()
            sd_dfs_d[pbn], hrs_d[pbn]['SD'] = calculate_single_dummy_probabilities(pbn, sd_productions) # all combinations of declarer pair direction, declarer direciton, suit, tricks taken
            if not progress and (i < 10 or i % 10000 == 0):
                print(f"calculate_single_dummy_probabilities: time:{time.time()-t} seconds")
            #error
    if progress:
        progress.progress(100,f"100%: Single dummies calculated for {len(unique_pbns)} of {len(unique_pbns)} unique deals using {sd_productions} samples per deal.")
    else:
        print(f"100%: Single dummies calculated for {len(unique_pbns)} of {len(unique_pbns)} unique deals using {sd_productions} samples per deal.")

    # create single dummy trick taking probability distribution columns
    sd_probs_d = defaultdict(list)
    for pbn in df['PBN']:
        productions, sd_d = hrs_d[pbn]['SD']
        for (pair_direction,declarer_direction,suit),probs in sd_d.items():
            #print(pair_direction,declarer_direction,suit)
            for i,t in enumerate(probs):
                sd_probs_d['_'.join(['Probs',pair_direction,declarer_direction,suit,str(i)])].append(t)
    # st.write(sd_probs_d)
    sd_probs_df = pl.DataFrame(sd_probs_d,orient='row')
    if progress:
        progress.empty()
    return sd_dfs_d, sd_probs_df


def create_scores_df_with_vul(scores_df):
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


def get_cached_sd_data(pbn, hrs_d):
    sd_data = hrs_d[pbn]['SD'][1]
    row_data = {'PBN': pbn}
    for (pair_direction, declarer_direction, strain), probs in sd_data.items():
        col_prefix = f"{pair_direction}_{declarer_direction}_{strain}"
        for i, prob in enumerate(probs):
            row_data[f"{col_prefix}_{i}"] = prob
    return row_data


def calculate_sd_expected_values(df, hrs_d, scores_df):

    # retrieve probabilities from cache
    sd_probs = [get_cached_sd_data(pbn, hrs_d) for pbn in df['PBN']]

    # Create a DataFrame from the extracted sd probs (frequency distribution of tricks).
    sd_df = pl.DataFrame(sd_probs)

    scores_df_vuls = create_scores_df_with_vul(scores_df)

    # Define the combinations
    # todo: move this to globals? but beware that globals can create weirdness with streamlit.
    pair_directions = ['NS', 'EW']
    declarer_directions = 'NESW'
    strains = 'SHDCN'
    levels = range(1,8)
    tricks = range(14)
    vuls = ['NV','V']

    # Perform the multiplication
    result = sd_df.select([
        pl.col(f'{pair_direction}_{declarer_direction}_{strain}_{taken}').mul(score).alias(f'{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}_{taken}_{score}')
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
    result = result.with_columns([
        pl.sum_horizontal(pl.col(f'^{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}_\\d+_.*$')).alias(f'EV_{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}')
        for pair_direction in pair_directions
        for declarer_direction in pair_direction #declarer_directions
        for strain in strains
        for level in levels
        for vul in vuls
    ])

    #print("\nResults with expected value:")
    return result


# calculate EV max scores for various regexes including all vulnerabilities. also create columns of the column names of the max values.
def create_best_contracts(df):

    # Define the combinations
    pair_directions = ['NS', 'EW']
    declarer_directions = 'NESW'
    strains = 'SHDCN'
    vulnerabilities = ['NV', 'V']

    # Function to create columns of max values from various regexes of columns. also creates columns of the column names of the max value.
    def max_and_col(df, pattern):
        cols = df.select(pl.col(pattern)).columns
        max_expr = pl.max_horizontal(pl.col(pattern))
        col_expr = pl.when(pl.col(cols[0]) == max_expr).then(pl.lit(cols[0]))
        for col in cols[1:]:
            col_expr = col_expr.when(pl.col(col) == max_expr).then(pl.lit(col))
        return max_expr, col_expr.otherwise(pl.lit(""))

    # Dictionary to store expressions with their aliases as keys
    max_ev_dict = {}

    # all EV columns are already calculated. just need to get the max.

    # Single loop handles all EV Max, Max_Col combinations
    for v in vulnerabilities:
        # Level 4: Overall Max EV for each vulnerability
        ev_columns = f'^EV_(NS|EW)_[NESW]_[SHDCN]_[1-7]_{v}$'
        max_expr, col_expr = max_and_col(df, ev_columns)
        max_ev_dict[f'EV_{v}_Max'] = max_expr
        max_ev_dict[f'EV_{v}_Max_Col'] = col_expr

        for pd in pair_directions:
            # Level 3: Max EV for each pair direction and vulnerability
            ev_columns = f'^EV_{pd}_[NESW]_[SHDCN]_[1-7]_{v}$'
            max_expr, col_expr = max_and_col(df, ev_columns)
            max_ev_dict[f'EV_{pd}_{v}_Max'] = max_expr
            max_ev_dict[f'EV_{pd}_{v}_Max_Col'] = col_expr

            for dd in pd: #declarer_directions:
                # Level 2: Max EV for each pair direction, declarer direction, and vulnerability
                ev_columns = f'^EV_{pd}_{dd}_[SHDCN]_[1-7]_{v}$'
                max_expr, col_expr = max_and_col(df, ev_columns)
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max'] = max_expr
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max_Col'] = col_expr

                for s in strains:
                    # Level 1: Max EV for each combination
                    ev_columns = f'^EV_{pd}_{dd}_{s}_[1-7]_{v}$'
                    max_expr, col_expr = max_and_col(df, ev_columns)
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


def convert_contract_to_contract(df):
    return df['Contract'].str.to_uppercase().str.replace('♠','S').str.replace('♥','H').str.replace('♦','D').str.replace('♣','C').str.replace('NT','N')


# None is used instead of pl.Null because pl.Null becomes 'Null' string in pl.String columns. Not sure what's going on but the solution is to use None.
def convert_contract_to_declarer(df):
    return [None if c is None or c == 'PASS' else c[2] for c in df['Contract']] # extract declarer from contract


def convert_contract_to_level(df):
    return [None if c is None or c == 'PASS' else c[0] for c in df['Contract']] # extract level from contract


def convert_contract_to_strain(df):
    return [None if c is None or c == 'PASS' else c[1] for c in df['Contract']] # extract strain from contract


def convert_contract_to_dbl(df):
    return [None if c is None or c == 'PASS' else c[3:] for c in df['Contract']] # extract dbl from contract


def convert_declarer_to_DeclarerName(df):
    return [None if d is None else df[d][i] for i,d in enumerate(df['Declarer_Direction'])] # extract declarer name using declarer direction as the lookup key


def convert_declarer_to_DeclarerID(df):
    return [None if d is None else df[f'Player_ID_{d}'][i] for i,d in enumerate(df['Declarer_Direction'])] # extract declarer name using declarer direction as the lookup key


def convert_contract_to_result(df):
    return [None if c is None or c == 'PASS' else 0 if c[-1] in ['=','0'] else int(c[-1]) if c[-2] == '+' else -int(c[-1]) for c in df['Contract']] # create result from contract


def convert_contract_to_tricks(df):
    return [None if c is None or c == 'PASS' else int(c[0])+6+r for c,r in zip(df['Contract'],df['Result'])] # create tricks from contract and result


def convert_contract_to_DD_Tricks(df):
    return [None if c is None or c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Declarer_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


def convert_contract_to_DD_Tricks_Dummy(df):
    return [None if c is None or c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Dummy_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


def convert_contract_to_DD_Score_Ref(df):
    # could use pl.str_concat() instead
    df = df.with_columns(
        (pl.lit('DD_Score_')+pl.col('BidLvl').cast(pl.String)+pl.col('BidSuit')+pl.lit('_')+pl.col('Declarer_Direction')).alias('DD_Score_Refs'),
    )
    ddscore_ns = []
    for i,(d,ref) in enumerate(zip(df['Declarer_Direction'],df['DD_Score_Refs'])):
        if ref is None:
            ddscore_ns.append(0)
        else:
            ddscore_ns.append(df[ref][i] if d in 'NS' else -df[ref][i])
    df = df.with_columns(pl.Series('DD_Score_NS',ddscore_ns,pl.Int16))
    df = df.with_columns(pl.col('DD_Score_NS').neg().alias('DD_Score_EW'))
    df = df.with_columns(pl.when(pl.col('Declarer_Direction').is_in(['N','S'])).then(pl.col('DD_Score_NS')).otherwise(pl.col('DD_Score_EW')).alias('DD_Score_Declarer'))
    return df

# additional augmentations for ACBL hand records
def AugmentACBLHandRecords(df,hrs_d):

    augmenter = HandAugmenter(df, hrs_d, sd_productions=40, progress=None)
    df = augmenter.perform_hand_augmentations()

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


def Perform_Legacy_Renames(df):

    df = df.with_columns([
        #pl.col('Section').alias('section_name'), # will this be needed for acbl?
        pl.col('N').alias('Player_Name_N'),
        pl.col('S').alias('Player_Name_S'),
        pl.col('E').alias('Player_Name_E'),
        pl.col('W').alias('Player_Name_W'),
        pl.col('Declarer_Name').alias('Name_Declarer'),
        pl.col('Declarer_ID').alias('Number_Declarer'), #  todo: rename to 'Declarer_ID'?
        pl.col('Declarer_Direction').replace_strict(mlBridgeLib.PlayerDirectionToPairDirection).alias('Declarer_Pair_Direction'),
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


def Create_Fake_Predictions(df):
    # todo: remove this once NN predictions are implemented
    df = df.with_columns(

        # pl.col('Pct_NS').alias('Pct_NS_Pred'),
        # pl.col('Pct_EW').alias('Pct_EW_Pred'),
        # pl.col('Pct_NS').sub(pl.col('Pct_NS')).alias('Pct_NS_Diff_Pred'),
        # pl.col('Pct_EW').sub(pl.col('Pct_EW')).alias('Pct_EW_Diff_Pred'),
        # pl.col('Declarer_Direction').alias('Declarer_Direction_Pred'), # Declarer_Direction_Actual not needed
        # pl.lit(.321).alias('Declarer_Pct_Pred'), # todo: implement 'Declarer_Pct'
        # pl.lit(456).alias('Declarer_Number_Pred'), # todo: implement 'Declarer_ID'
        # pl.col('Declarer_Name').alias('Declarer_Name_Pred'),
        # pl.col('Contract').alias('Contract_Pred'),
    )
    return df


# def calculate_matchpoint_scores_ns(df,score_columns):
    
#     # Process each row
#     mp_columns = defaultdict(list)
#     for r in df.iter_rows(named=True):
#         scores_list = r['Expanded_Scores_List'] # todo: make 'Expanded_Scores_List' sorted and with a 'Score_NS' removed?
#         if scores_list is None:
#             # todo: kludge: apparently Expanded_Scores_List can be null if director's adjustment.
#             # matchpoints can't be computed because there's no list of scores. we only have scores at player's table, not all tables.
#             # The fix is to download all table's results to replace a null Expanded_Scores_List.
#             print(f"Expanded_Scores_List is null: Score_NS:{r['Score_NS']} MP_NS:{r['MP_NS']}. Skipping.")
#             for col in score_columns:
#                 mp_columns['MP_'+col].append(0.5)
#             continue
#         scores_list.remove(r['Score_NS'])
        
#         for col in score_columns:
#             # Calculate rank for each DD score
#             rank = 0.0
#             new_score = r[col]
#             if scores_list:
#                 for score in scores_list:
#                     if new_score > score:
#                         rank += 1.0
#                     elif new_score == score:
#                         rank += 0.5
                    
#             mp_columns['MP_'+col].append(rank)
    
#     # Add all new columns at once
#     return df.hstack(pl.DataFrame(mp_columns))


def DealToCards(df):
    lazy_df = df.lazy()
    lazy_cards_df = lazy_df.with_columns([
        pl.col(f'Suit_{direction}_{suit}').str.contains(rank).alias(f'C_{direction}{suit}{rank}')
        for direction in 'NESW'
        for suit in 'SHDC'
        for rank in 'AKQJT98765432'
    ])
    return lazy_cards_df.collect()


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
    return df.with_columns(partnership_qt)


def calculate_LoTT(df):

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


class HandAugmenter:
    def __init__(self, df, hrs_d=None, sd_productions=40, progress=None):
        self.df = df
        self.hrs_d = hrs_d if hrs_d is not None else {}
        self.sd_productions = sd_productions
        self.progress = progress
        self.declarer_to_LHO_d = {None:None,'N':'E','E':'S','S':'W','W':'N'}
        self.declarer_to_dummy_d = {None:None,'N':'S','E':'W','S':'N','W':'E'}
        self.declarer_to_RHO_d = {None:None,'N':'W','E':'N','S':'E','W':'S'}
        self.vul_conditions = {
            'NS': pl.col('Vul').is_in(['N_S', 'Both']),
            'EW': pl.col('Vul').is_in(['E_W', 'Both'])
        }

    def _time_operation(self, operation_name, func, *args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _add_default_columns(self):
        if 'group_id' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit(0).alias('group_id'))
        if 'session_id' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit(0).alias('session_id'))
        if 'section_name' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit('').alias('section_name'))

    def _create_hand_columns(self):
        self.df = self._time_operation("create_hand_nesw_columns", create_hand_nesw_columns, self.df)
        self.df = self._time_operation("create_suit_nesw_columns", create_suit_nesw_columns, self.df)
        self.df = self._time_operation("create_hands_lists_column", create_hands_lists_column, self.df)

    def _process_scores_and_tricks(self):
        all_scores_d, scores_d, scores_df = self._time_operation("calculate_scores", calculate_scores)
        DD_Tricks_df, par_df, dd_score_df = self._time_operation(
            "calculate_ddtricks_par_scores", 
            calculate_ddtricks_par_scores, 
            self.df, self.hrs_d, scores_d, progress=self.progress
        )
        sd_dfs_d, sd_probs_df = self._time_operation(
            "calculate_sd_probs",
            calculate_sd_probs,
            self.df, self.hrs_d, self.sd_productions, self.progress
        )
        sd_ev_df = self._time_operation(
            "calculate_sd_expected_values",
            calculate_sd_expected_values,
            self.df, self.hrs_d, scores_df
        )
        best_contracts_df = self._time_operation("create_best_contracts", create_best_contracts, sd_ev_df)
        
        self.df = pl.concat(
            [self.df, DD_Tricks_df, par_df, dd_score_df, sd_probs_df, sd_ev_df, best_contracts_df],
            how='horizontal'
        )
        return scores_df

    def _process_contract_columns(self):
        if 'Player_Name_N' in self.df.columns:
            self.df = self._time_operation(
                "rename_players",
                lambda df: df.rename({'Player_Name_N':'N','Player_Name_E':'E','Player_Name_S':'S','Player_Name_W':'W'}),
                self.df
            )

        self.df = self._time_operation(
            "convert_contract_to_contract",
            lambda df: df.with_columns(
                pl.Series('Contract', convert_contract_to_contract(df), pl.String, strict=False)
            ),
            self.df
        )

        self.df = self._time_operation(
            "convert_contract_parts",
            lambda df: df.with_columns([
                pl.Series('Declarer_Direction', convert_contract_to_declarer(df), pl.String, strict=False),
                pl.Series('BidLvl', convert_contract_to_level(df), pl.UInt8, strict=False),
                pl.Series('BidSuit', convert_contract_to_strain(df), pl.String, strict=False),
                pl.Series('Dbl', convert_contract_to_dbl(df), pl.String, strict=False),
            ]),
            self.df
        )

    def _create_contract_types(self):
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

    def _create_direction_columns(self):
        self.df = self._time_operation(
            "convert_contract_to_directions",
            lambda df: df.with_columns([
                pl.col('Declarer_Direction').replace_strict(self.declarer_to_LHO_d).alias('LHO_Direction'),
                pl.col('Declarer_Direction').replace_strict(self.declarer_to_dummy_d).alias('Dummy_Direction'),
                pl.col('Declarer_Direction').replace_strict(self.declarer_to_RHO_d).alias('RHO_Direction'),
            ]),
            self.df
        )

    def _create_declarer_columns(self):
        self.df = self._time_operation(
            "convert_declarer_columns",
            lambda df: df.with_columns([
                pl.Series('Declarer_Name', convert_declarer_to_DeclarerName(df), pl.String, strict=False),
                pl.Series('Declarer_ID', convert_declarer_to_DeclarerID(df), pl.String, strict=False),
            ]),
            self.df
        )

    def _create_result_columns(self):
        if 'Result' not in self.df.columns:
            assert 'Contract' in self.df.columns, 'Contract column is required to create Result column.'
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

    def _create_dd_columns(self):
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

    def _create_score_columns(self):
        if 'Score_NS' not in self.df.columns:
            self.df = self._time_operation(
                "convert_score_to_score",
                lambda df: df.with_columns([
                    pl.col('Score').alias('Score_NS'),
                    pl.col('Score').neg().alias('Score_EW')
                ]),
                self.df
            )

    def _create_ev_columns(self):
        max_expressions = []
        for pd in ['NS', 'EW']:
            max_expressions.extend(self._create_ev_expressions_for_pair(pd))
        
        self.df = self._time_operation(
            "create_ev_columns",
            lambda df: df.with_columns(max_expressions).with_columns([
                pl.max_horizontal('EV_Max_NS','EV_Max_EW').alias('EV_Max'),
                pl.max_horizontal('EV_Max_Col_NS','EV_Max_Col_EW').alias('EV_Max_Col'),
                pl.when(pl.col('Declarer_Direction').is_in(['N','S'])).then(pl.col('EV_Max_NS')).otherwise(pl.col('EV_Max_EW')).alias('EV_Max_Declarer'),
            ]),
            self.df
        )

    def _create_ev_expressions_for_pair(self, pd):
        expressions = []
        expressions.extend(self._create_basic_ev_expressions(pd))
        
        for dd in pd:
            expressions.extend(self._create_declarer_ev_expressions(pd, dd))
            
            for s in 'SHDCN':
                expressions.extend(self._create_strain_ev_expressions(pd, dd, s))
                
                for l in range(1, 8):
                    expressions.extend(self._create_level_ev_expressions(pd, dd, s, l))
        
        return expressions

    def _create_basic_ev_expressions(self, pd):
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

    def _create_declarer_ev_expressions(self, pd, dd):
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

    def _create_strain_ev_expressions(self, pd, dd, s):
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

    def _create_level_ev_expressions(self, pd, dd, s, l):
        return [
            pl.when(self.vul_conditions[pd])
            .then(pl.col(f'EV_{pd}_{dd}_{s}_{l}_V'))
            .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_{l}_NV'))
            .alias(f'EV_{pd}_{dd}_{s}_{l}')
        ]

    def _create_diff_columns(self):
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

    def perform_hand_augmentations(self):
        """Main method to perform all hand augmentations"""
        self._add_default_columns()
        self._create_hand_columns()
        scores_df = self._process_scores_and_tricks()
        self._process_contract_columns()
        self._create_contract_types()
        self._create_direction_columns()
        self._create_declarer_columns()
        self._create_result_columns()
        self._create_dd_columns()
        self._create_score_columns()
        self._create_ev_columns()
        self._create_diff_columns()
        return self.df

class ResultAugmenter:
    def __init__(self, df, hrs_d=None):
        self.df = df
        self.hrs_d = hrs_d if hrs_d is not None else {}
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

    def _time_operation(self, operation_name, func, *args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_cards(self):
        if 'C_NSA' not in self.df.columns:
            self.df = self._time_operation("create C_NSA", DealToCards, self.df)

    def _create_hcp(self):
        if 'HCP_N_C' not in self.df.columns:
            self.df = self._time_operation("create HCP", CardsToHCP, self.df)

    def _create_quick_tricks(self):
        if 'QT_N_C' not in self.df.columns:
            self.df = self._time_operation("create QT", CardsToQuickTricks, self.df)

    def _create_suit_lengths(self):
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

    def _create_pair_suit_lengths(self):
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

    def _create_suit_length_arrays(self):
        if 'SL_N_CDHS' not in self.df.columns:
            def create_sl_arrays(df, direction):
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

    def _create_distribution_points(self):
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
                ]),
                self.df
            )

    def _create_total_points(self):
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

    def _create_max_suit_lengths(self):
        if 'SL_Max_NS' not in self.df.columns:
            sl_cols = [('_'.join(['SL_Max',d]), ['_'.join(['SL',d,s]) for s in mlBridgeLib.SHDC]) 
                      for d in mlBridgeLib.NS_EW]
            for d in sl_cols:
                self.df = self._time_operation(
                    f"create {d[0]}",
                    lambda df: df.with_columns(
                        pl.Series(d[0], [d[1][l.index(max(l))] for l in df[d[1]].rows()])
                    ),
                    self.df
                )

    def _create_lott(self):
        if 'LoTT' not in self.df.columns:
            self.df = self._time_operation("create LoTT", calculate_LoTT, self.df)

    def _create_contract_types(self):
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

    def _create_contract_type_booleans(self):
        if 'CT_N_C_Game' not in self.df.columns:
            ct_boolean_columns = [
                pl.col(f"CT_{direction}_{strain}").eq(pl.lit(contract))
                .alias(f"CT_{direction}_{strain}_{contract}")
                for direction in "NESW"
                for strain in "SHDCN"
                for contract in ["Pass","Game","SSlam","GSlam","Partial"]
            ]
            self.df = self._time_operation(
                "create CT boolean columns",
                lambda df: df.with_columns(ct_boolean_columns),
                self.df
            )

    def _create_dealer(self):
        if 'Dealer' not in self.df.columns:
            def board_number_to_dealer(bn):
                return 'NESW'[(bn-1) & 3]
            
            self.df = self._time_operation(
                "create Dealer",
                lambda df: df.with_columns(
                    pl.col('board_boardNumber')
                    .map_elements(board_number_to_dealer, return_dtype=pl.String)
                    .alias('Dealer')
                ),
                self.df
            )

    def _create_vulnerability(self):
        if 'iVul' not in self.df.columns:
            if 'Vul' in self.df.columns:
                def vul_to_ivul(vul):
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
                def board_number_to_vul(bn):
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
            def ivul_to_vul(ivul):
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

    def _create_quality_indicators(self):
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

    def _create_balanced_indicators(self):
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

    def perform_result_augmentations(self):
        """Main method to perform all result augmentations"""
        self._create_cards()
        self._create_hcp()
        self._create_quick_tricks()
        self._create_suit_lengths()
        self._create_pair_suit_lengths()
        self._create_suit_length_arrays()
        self._create_distribution_points()
        self._create_total_points()
        self._create_max_suit_lengths()
        self._create_lott()
        self._create_contract_types()
        self._create_contract_type_booleans()
        self._create_dealer()
        self._create_vulnerability()
        self._create_quality_indicators()
        self._create_balanced_indicators()
        return self.df

class DDSDAugmenter:
    def __init__(self, df):
        self.df = df

    def _time_operation(self, operation_name, func, *args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _perform_legacy_renames(self):
        self.df = self._time_operation(
            "perform legacy renames",
            Perform_Legacy_Renames,
            self.df
        )

    def _create_fake_predictions(self):
        self.df = self._time_operation(
            "create fake predictions",
            Create_Fake_Predictions,
            self.df
        )

    def _create_declarer_columns(self):
        self.df = self._time_operation(
            "create declarer columns",
            lambda df: df.with_columns([
                pl.concat_str([
                    pl.lit('EV'),
                    pl.col('Declarer_Pair_Direction'),
                    pl.col('Declarer_Direction'),
                    pl.col('BidSuit'),
                    pl.col('BidLvl').cast(pl.String),
                ], separator='_').alias('EV_Score_Col_Declarer'),
                
                pl.when(pl.col('Declarer_Pair_Direction').eq(pl.lit('NS')))
                .then(pl.col('Score_NS'))
                .otherwise(pl.col('Score_EW'))
                .alias('Score_Declarer'),
                
                pl.when(pl.col('Declarer_Pair_Direction').eq(pl.lit('NS')))
                .then(pl.col('Par_NS'))
                .otherwise(pl.col('Par_EW'))
                .alias('Par_Declarer'),
                
                ((pl.col('Declarer_Pair_Direction').eq('NS') & pl.col('Vul_NS')) | 
                 (pl.col('Declarer_Pair_Direction').eq('EW') & pl.col('Vul_EW')))
                .alias('Declarer_Vul'),
                
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
            ]),
            self.df
        )

    def _create_position_columns(self):
        self.df = self._time_operation(
            "create position columns",
            lambda df: df.with_columns([
                pl.col('Declarer_Direction').replace_strict(mlBridgeLib.NextPosition).alias('Direction_OnLead'),
            ])
            .with_columns([
                pl.col('Declarer_Pair_Direction').replace_strict(mlBridgeLib.PairDirectionToOpponentPairDirection).alias('Opponent_Pair_Direction'),
                pl.struct(['Declarer_Pair_Direction', 'Score_NS', 'Score_EW']).map_elements(
                    lambda r: None if r['Declarer_Pair_Direction'] is None else r[f'Score_{r["Declarer_Pair_Direction"]}'],
                    return_dtype=pl.Int16
                ).alias('Score_Declarer'),
                pl.col('Direction_OnLead').replace_strict(mlBridgeLib.NextPosition).alias('Direction_Dummy'),
                pl.struct(['Direction_OnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_OnLead'] is None else r[f'Player_ID_{r["Direction_OnLead"]}'],
                    return_dtype=pl.String
                ).alias('OnLead'),
            ])
            .with_columns([
                pl.col('Direction_Dummy').replace_strict(mlBridgeLib.NextPosition).alias('Direction_NotOnLead'),
                pl.struct(['Direction_Dummy', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_Dummy'] is None else r[f'Player_ID_{r["Direction_Dummy"]}'],
                    return_dtype=pl.String
                ).alias('Dummy'),
                pl.col('Score_Declarer').le(pl.col('Par_Declarer')).alias('Defender_Par_GE')
            ]),
            self.df
        )

    def _create_additional_columns(self):
        self.df = self._time_operation(
            "create additional columns",
            lambda df: df.with_columns([
                pl.struct(['Direction_NotOnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_NotOnLead'] is None else r[f"Player_ID_{r["Direction_NotOnLead"]}"],
                    return_dtype=pl.String
                ).alias('NotOnLead'),
                pl.struct(['Declarer_Pair_Direction', 'Vul_NS', 'Vul_EW']).map_elements(
                    lambda r: None if r['Declarer_Pair_Direction'] is None else r[f'Vul_{r["Declarer_Pair_Direction"]}'],
                    return_dtype=pl.Boolean
                ).alias('Vul_Declarer'),
                # ... (remaining struct mappings)
            ]),
            self.df
        )

    def _create_board_result_columns(self):
        print(self.df.filter(pl.col('Result').is_null() | pl.col('Tricks').is_null())
              ['Contract','Declarer_Direction','Declarer_Vul','Vul_Declarer','iVul','Score_NS','BidLvl','Result','Tricks'])
        
        all_scores_d, scores_d, scores_df = calculate_scores()
        
        self.df = self._time_operation(
            "create board result columns",
            lambda df: df.with_columns([
                pl.struct(['EV_Score_Col_Declarer','^EV_(NS|EW)_[NESW]_[SHDCN]_[1-7]$'])
                    .map_elements(lambda x: None if x['EV_Score_Col_Declarer'] is None else x[x['EV_Score_Col_Declarer']],
                                return_dtype=pl.Float32).alias('EV_Score_Declarer'),
                pl.struct(['BidLvl', 'BidSuit', 'Tricks', 'Vul_Declarer', 'Dbl'])
                    .map_elements(lambda x: all_scores_d.get(tuple(x.values()),None),
                                return_dtype=pl.Int16)
                    .alias('Computed_Score_Declarer'),

                pl.struct(['Contract', 'Result', 'Score_NS', 'BidLvl', 'BidSuit', 'Dbl','Declarer_Direction', 'Vul_Declarer']).map_elements(
                    lambda r: None if r['Contract'] is None else 0 if r['Contract'] == 'PASS' else r['Score_NS'] if r['Result'] is None else mlBridgeLib.score(
                        r['BidLvl'] - 1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), 'NESW'.index(r['Declarer_Direction']),
                        r['Vul_Declarer'], r['Result'], True),return_dtype=pl.Int16).alias('Computed_Score_Declarer2'),
            ]),
            self.df
        )
        # todo: can remove df['Computed_Score_Declarer2'] after assert has proven equality
        # if asserts, may be due to Result or Tricks having nulls.
        assert self.df['Computed_Score_Declarer'].eq(self.df['Computed_Score_Declarer2']).all()

    def _create_trick_columns(self):
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

    def _create_rating_columns(self):
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

    def perform_dd_sd_augmentations(self):
        """Main method to perform all DD and SD augmentations"""
        self._perform_legacy_renames()
        self._create_fake_predictions()
        self._create_declarer_columns()
        self._create_position_columns()  # This is failing because it needs Pair_Declarer_Direction
        self._create_additional_columns()
        self._create_board_result_columns()
        self._create_trick_columns()
        self._create_rating_columns()
        return self.df


class MatchPointAugmenter:
    def __init__(self, df):
        self.df = df
        self.discrete_score_columns = [] # ['DD_Score_NS', 'EV_Max_NS'] # calculate matchpoints for these columns which change with each row's Score_NS
        self.dd_score_columns = [f'DD_Score_{l}{s}_{d}' for d in 'NESW' for s in 'SHDCN' for l in range(1,8)]
        self.ev_score_columns = [f'EV_{pd}_{d}_{s}_{l}' for pd in ['NS','EW'] for d in pd for s in 'SHDCN' for l in range(1,8)]
        self.all_score_columns = self.discrete_score_columns + self.dd_score_columns + self.ev_score_columns

    def _time_operation(self, operation_name, func, *args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_mp_top(self):
        if 'MP_Top' not in self.df.columns:
            self.df = self._time_operation(
                "create MP_Top",
                lambda df: df.with_columns(
                    pl.col('Score').count().over(['session_id','PBN','Board']).sub(1).alias('MP_Top')
                ),
                self.df
            )

    def _calculate_matchpoints(self):
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

    def _calculate_percentages(self):
        if 'Pct_NS' not in self.df.columns:
            self.df = self._time_operation(
                "calculate matchpoints percentages",
                lambda df: df.with_columns([
                    (pl.col('MP_NS') / pl.col('MP_Top')).alias('Pct_NS'),
                    (pl.col('MP_EW') / pl.col('MP_Top')).alias('Pct_EW')
                ]),
                self.df
            )

    def _create_declarer_pct(self):
        if 'Declarer_Pct' not in self.df.columns:
            self.df = self._time_operation(
                "create Declarer_Pct",
                lambda df: df.with_columns(
                    pl.when(pl.col('Declarer_Direction').is_in(['N','S']))
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
        score_ns_values = score_ns_values.fill_null(0.0) # todo: why do some have nulls? sitout?
        col_values = col_values.fill_null(0.0) # todo: why do some have nulls? sitout?
        return pl.Series([
            sum(1.0 if val > score else 0.5 if val == score else 0.0 
                for score in score_ns_values)
            for val in col_values
        ])


    def _calculate_all_score_matchpoints(self):
        t = time.time()
        # if 'Expanded_Scores_List' in self.df.columns: # todo: obsolete?
        #     print('Calculate matchpoints for existing Expanded_Scores_List column.')
        #     self.df = calculate_matchpoint_scores_ns(self.df, self.all_score_columns)
        # else:
        print('Calculate matchpoints over session, PBN, and Board.')
        # calc matchpoints on row-by-row basis
        if self.df['Score_NS'].is_null().sum() > 0:
            print(f"Warning: Null values in score_ns_values: {self.df['Score_NS'].is_null().sum()}")
        # for NS scores
        for col in self.all_score_columns + ['DD_Score_NS', 'Par_NS']:
            assert 'MP_'+col not in self.df.columns, f"Column 'MP_{col}' already exists in DataFrame"
            self.df = self.df.with_columns([
                    pl.map_groups(
                        exprs=[col, 'Score_NS'],
                        function=self._calculate_matchpoints_group,
                        return_dtype=pl.Float64,
                    ).over(['session_id', 'PBN', 'Board']).alias('MP_'+col)
                ])
        # for declarer orientation scores
        for col in [('DD_Score_Declarer','Score_Declarer'),('Par_Declarer','Score_Declarer'),('EV_Score_Declarer','Score_Declarer'),('EV_Max_Declarer','Score_Declarer')]:
            assert 'MP_'+col[0] not in self.df.columns, f"Column 'MP_{col[0]}' already exists in DataFrame"
            self.df = self.df.with_columns([
                    pl.map_groups(
                        exprs=col,
                        function=self._calculate_matchpoints_group,
                        return_dtype=pl.Float64,
                    ).over(['session_id', 'PBN', 'Board']).alias('MP_'+col[0])
                ])
        print(f"calculate matchpoints all_score_columns: time:{time.time()-t} seconds")

    def _calculate_final_scores(self):
        t = time.time()
        
        # Calculate MP and percentages for discrete scores
        for col_ns in ['DD_Score_NS','Par_NS']:
            col_ew = col_ns.replace('NS','EW')
            self.df = self.df.with_columns(
                (pl.col('MP_Top')-pl.col(f'MP_{col_ns}')).alias(f'MP_{col_ew}')
            ).with_columns([
                (pl.col(f'MP_{col_ns}')/pl.col('MP_Top')).alias(col_ns.replace('_NS','_Pct_NS')),
                (pl.col(f'MP_{col_ew}')/pl.col('MP_Top')).alias(col_ew.replace('_EW','_Pct_EW')),
            ])
        # for declarer orientation scores
        self.df = self.df.with_columns(
            (pl.col(f'MP_DD_Score_Declarer')/pl.col('MP_Top')).alias('MP_DD_Pct_Declarer'),
            (pl.col(f'MP_Par_Declarer')/pl.col('MP_Top')).alias('MP_Par_Pct_Declarer'),
            (pl.col(f'MP_EV_Score_Declarer')/pl.col('MP_Top')).alias('MP_EV_Pct_Declarer'),
            (pl.col(f'MP_EV_Max_Declarer')/pl.col('MP_Top')).alias('MP_EV_Max_Pct_Declarer')
        )

        # Calculate remaining scores and percentages
        operations = [
            lambda df: df.with_columns((1-pl.col('Par_Pct_NS')).alias('Par_Pct_EW')),
            lambda df: df.with_columns(pl.max_horizontal(f'^MP_DD_Score_[1-7][SHDCN]_[NS]$').alias(f'MP_DD_Score_NS_Max')),
            lambda df: df.with_columns(pl.max_horizontal(f'^MP_DD_Score_[1-7][SHDCN]_[EW]$').alias(f'MP_DD_Score_EW_Max')),
            lambda df: df.with_columns(pl.max_horizontal(f'^MP_EV_NS_[NS]_[SHDCN]_[1-7]$').alias(f'MP_EV_Max_NS')),
            lambda df: df.with_columns(pl.max_horizontal(f'^MP_EV_EW_[EW]_[SHDCN]_[1-7]$').alias(f'MP_EV_Max_EW')),
            lambda df: df.with_columns([
                (pl.col('MP_DD_Score_NS_Max')/pl.col('MP_Top')).alias('DD_Score_Pct_NS_Max'),
                (pl.col('MP_DD_Score_EW_Max')/pl.col('MP_Top')).alias('DD_Score_Pct_EW_Max'),
                (pl.col('MP_EV_Max_NS')/pl.col('MP_Top')).alias('EV_Pct_Max_NS'),
                (pl.col('MP_EV_Max_EW')/pl.col('MP_Top')).alias('EV_Pct_Max_EW'),
                #pl.col('DD_Score_Pct_NS').alias('DD_Pct_NS'),
                #pl.col('DD_Score_Pct_EW').alias('DD_Pct_EW'),
                #pl.col('MP_NS').alias('Matchpoints_NS'),
                #pl.col('MP_EW').alias('Matchpoints_EW'),
                #pl.col('MP_EV_Max_NS').alias('SD_MP_Max_NS'),
                #pl.col('MP_Top').sub(pl.col('MP_EV_Max_NS')).alias('SD_MP_Max_EW'),
            ]),
            lambda df: df.with_columns([
                #pl.col('EV_Pct_Max_NS').alias('SD_Pct_NS'),
                #pl.col('EV_Pct_Max_EW').alias('SD_Pct_EW'),
                #pl.col('EV_Pct_Max_NS').alias('SD_Pct_Max_NS'),
                #pl.col('EV_Pct_Max_EW').alias('SD_Pct_Max_EW'),
                (pl.col('EV_Pct_Max_NS')-pl.col('Pct_NS')).alias('EV_Pct_Max_Diff_NS'),
                (pl.col('EV_Pct_Max_EW')-pl.col('Pct_EW')).alias('EV_Pct_Max_Diff_EW'),
                (pl.col('Par_Pct_NS')-pl.col('Pct_NS')).alias('EV_Par_Pct_Diff_NS'),
                (pl.col('Par_Pct_EW')-pl.col('Pct_EW')).alias('EV_Par_Pct_Diff_EW'),
                (pl.col('Par_Pct_NS')-pl.col('Pct_NS')).alias('EV_Par_Pct_Max_Diff_NS'),
                (pl.col('Par_Pct_EW')-pl.col('Pct_EW')).alias('EV_Par_Pct_Max_Diff_EW'),
            ])
        ]

        for operation in operations:
            self.df = operation(self.df)

        print(f"Time to rank expanded scores: {time.time()-t} seconds")

    def perform_matchpoint_augmentations(self):
        self._create_mp_top()
        self._calculate_matchpoints()
        self._calculate_percentages()
        self._create_declarer_pct()
        self._calculate_all_score_matchpoints()
        self._calculate_final_scores()
        return self.df
