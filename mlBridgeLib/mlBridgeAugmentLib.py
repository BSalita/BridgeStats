# todo:
# we're reading just one url which contains only the player's results. Need to read all urls to get all board results.
# rename columns

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


# global variables static and read-only
# oops. globals don't behave as expected in streamlit. need to use st.session_state but that's not available in this module. just recompute.
#scores_d = None # (level,suit_char,tricks,vul) -> score
#all_scores_d = None # (level,suit_char,tricks,vul,dbl) -> score
#scores_df = None # 'Score_[1-7][SHDCN]'

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
                progress.progress(percent_complete,f"{percent_complete}%: {i} of {len(deals)} double dummies calculated.")
            else:
                if i % 1000 == 0:
                    percent_complete = int(i*100/len(deals))
                    print(f"{percent_complete}%: {i} of {len(deals)} double dummies calculated")
        result_tables = calc_all_tables(deals[b:b+batch_size])
        all_result_tables.extend(result_tables)
    if output_progress: 
        if progress:
            progress.progress(100,f"100%: {len(deals)} of {len(deals)} double dummies calculated.")
        else:
            print(f"100%: {len(deals)} of {len(deals)} double dummies calculated.")
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
    par_df = pl.DataFrame({'ParScore_NS': par_scores_ns, 'ParScore_EW': par_scores_ew, 'ParContract': par_contracts},orient='row')

    # Create column names
    columns = {f'DD_{direction}_{suit}':pl.UInt8 for direction in 'NESW' for suit in 'SHDCN'}

    # Create the DataFrame
    DDTricks_df = pl.DataFrame(flattened_dd_rows, schema=columns, orient='row')

    dd_ns_ew_columns = [
        pl.max_horizontal(f"DD_{pair[0]}_{strain}",f"DD_{pair[1]}_{strain}").alias(f"DD_{pair}_{strain}")
        for pair in ['NS','EW']
        for strain in "SHDCN"
    ]
    DDTricks_df = DDTricks_df.with_columns(dd_ns_ew_columns)

    dd_score_cols = [[scores_d[(level,suit,tricks,vul == 'Both' or (vul != 'None' and direction in vul))] for tricks,vul in zip(DDTricks_df['_'.join(['DD',direction,suit])],df['Vul'])] for direction in 'NESW' for suit in 'SHDCN' for level in range(1, 8)]
    dd_score_df = pl.DataFrame(dd_score_cols, schema=['_'.join(['DDScore', str(l) + s, d]) for d in 'NESW' for s in 'SHDCN' for l in range(1, 8)])
    
    return DDTricks_df, par_df, dd_score_df


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
        SDTricks_df = pl.DataFrame([[sddeal.to_pbn()]+[s for d in t.to_list() for s in d] for sddeal,t in zip(sd_deals,sd_dd_result_tables)],schema={'SD_Deal':pl.String}|{'_'.join(['SDTricks',d,s]):pl.UInt8 for d in 'NESW' for s in 'SHDCN'},orient='row')

        for d in 'NSEW':
            for s in 'SHDCN':
                # always create 14 rows (0-13 tricks taken) for combo of direction and suit. fill never-happened with proper index and 0.0 prob value.
                #ns_ew_rows[(ns_ew,d,s)] = dd_df[d+s].to_pandas().value_counts(normalize=True).reindex(range(14), fill_value=0).tolist() # ['Fixed_Direction','Direction_Declarer','Suit']+['SD_Prob_Take_'+str(n) for n in range(14)]
                vc = {ds:p for ds,p in SDTricks_df['_'.join(['SDTricks',d,s])].value_counts(normalize=True).rows()}
                index = {i:0.0 for i in range(14)} # fill values for missing probs
                ns_ew_rows[(ns_ew,d,s)] = list((index|vc).values())

    return SDTricks_df, (produce, ns_ew_rows)


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
            progress.progress(percent_complete,f"{percent_complete}%: {i} of {len(unique_pbns)} single dummies calculated using {sd_productions} samples")
        else:
            if i < 10 or i % 10000 == 0:
                percent_complete = int(i*100/len(unique_pbns))
                print(f"{percent_complete}%: {i} of {len(unique_pbns)} single dummies calculated using {sd_productions} samples")
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
        progress.progress(100,f"100%: {len(unique_pbns)} of {len(unique_pbns)} single dummies calculated.")
    else:
        print(f"100%: {len(unique_pbns)} of {len(unique_pbns)} single dummies calculated.")

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

    # Single loop handles all EV Max, MaxCol combinations
    for v in vulnerabilities:
        # Level 4: Overall Max EV for each vulnerability
        ev_columns = f'^EV_(NS|EW)_[NESW]_[SHDCN]_[1-7]_{v}$'
        max_expr, col_expr = max_and_col(df, ev_columns)
        max_ev_dict[f'EV_{v}_Max'] = max_expr
        max_ev_dict[f'EV_{v}_MaxCol'] = col_expr

        for pd in pair_directions:
            # Level 3: Max EV for each pair direction and vulnerability
            ev_columns = f'^EV_{pd}_[NESW]_[SHDCN]_[1-7]_{v}$'
            max_expr, col_expr = max_and_col(df, ev_columns)
            max_ev_dict[f'EV_{pd}_{v}_Max'] = max_expr
            max_ev_dict[f'EV_{pd}_{v}_MaxCol'] = col_expr

            for dd in pd: #declarer_directions:
                # Level 2: Max EV for each pair direction, declarer direction, and vulnerability
                ev_columns = f'^EV_{pd}_{dd}_[SHDCN]_[1-7]_{v}$'
                max_expr, col_expr = max_and_col(df, ev_columns)
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max'] = max_expr
                max_ev_dict[f'EV_{pd}_{dd}_{v}_MaxCol'] = col_expr

                for s in strains:
                    # Level 1: Max EV for each combination
                    ev_columns = f'^EV_{pd}_{dd}_{s}_[1-7]_{v}$'
                    max_expr, col_expr = max_and_col(df, ev_columns)
                    max_ev_dict[f'EV_{pd}_{dd}_{s}_{v}_Max'] = max_expr
                    max_ev_dict[f'EV_{pd}_{dd}_{s}_{v}_MaxCol'] = col_expr

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
    return [None if c == 'PASS' else c[2] for c in df['Contract']] # extract declarer from contract


def convert_contract_to_level(df):
    return [None if c == 'PASS' else c[0] for c in df['Contract']] # extract level from contract


def convert_contract_to_strain(df):
    return [None if c == 'PASS' else c[1] for c in df['Contract']] # extract strain from contract


def convert_contract_to_dbl(df):
    return [None if c == 'PASS' else c[3:] for c in df['Contract']] # extract dbl from contract


def convert_declarer_to_DeclarerName(df):
    return [None if d is None else df[d][i] for i,d in enumerate(df['Declarer_Direction'])] # extract declarer name using declarer direction as the lookup key


def convert_declarer_to_DeclarerID(df):
    return [None if d is None else df[f'Player_ID_{d}'][i] for i,d in enumerate(df['Declarer_Direction'])] # extract declarer name using declarer direction as the lookup key


def convert_contract_to_result(df):
    return [None if c == 'PASS' else 0 if c[-1] in ['=','0'] else int(c[-1]) if c[-2] == '+' else -int(c[-1]) for c in df['Contract']] # create result from contract


def convert_contract_to_tricks(df):
    return [None if c == 'PASS' else int(c[0])+6+r for c,r in zip(df['Contract'],df['Result'])] # create tricks from contract and result


def convert_contract_to_DDTricks(df):
    return [None if c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Declarer_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


def convert_contract_to_DDTricks_Dummy(df):
    return [None if c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Dummy_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


def convert_contract_to_DDScore_Ref(df):
    # could use pl.str_concat() instead
    df = df.with_columns(
        (pl.lit('DDScore_')+pl.col('BidLvl').cast(pl.String)+pl.col('BidSuit')+pl.lit('_')+pl.col('Declarer_Direction')).alias('DDScore_Refs'),
    )
    ddscore_ns = []
    for i,(d,ref) in enumerate(zip(df['Declarer_Direction'],df['DDScore_Refs'])):
        if ref is None:
            ddscore_ns.append(0)
        else:
            ddscore_ns.append(df[ref][i] if d in 'NS' else -df[ref][i])
    df = df.with_columns(pl.Series('DDScore_NS',ddscore_ns,pl.Int16))
    df = df.with_columns(pl.col('DDScore_NS').neg().alias('DDScore_EW'))
    return df


def perform_hand_augmentations(df,hrs_d,sd_productions=40,progress=None):

    # todo: refactor all of these df ops into separate functions.

    t = time.time()
    if 'group_id' not in df.columns:
        df = df.with_columns(
            pl.lit(0).alias('group_id')
        )
    print(f"create group_id: time:{time.time()-t} seconds")

    t = time.time()
    if 'session_id' not in df.columns:
        df = df.with_columns(
            pl.lit(0).alias('session_id')
        )
    print(f"create session_id: time:{time.time()-t} seconds")

    t = time.time()
    if 'section_name' not in df.columns:
        df = df.with_columns(
            pl.lit('').alias('section_name')
        )
    print(f"create section_name: time:{time.time()-t} seconds")

    t = time.time()
    df = create_hand_nesw_columns(df)
    print(f"create_hand_nesw_columns: time:{time.time()-t} seconds")

    t = time.time()
    df = create_suit_nesw_columns(df)
    print(f"create_suit_nesw_columns: time:{time.time()-t} seconds")

    t = time.time()
    df = create_hands_lists_column(df)
    print(f"create_hands_lists_column: time:{time.time()-t} seconds")

    t = time.time()
    all_scores_d, scores_d, scores_df = calculate_scores()
    print(f"calculate_scores: time:{time.time()-t} seconds")

    t = time.time()
    DDTricks_df, par_df, dd_score_df = calculate_ddtricks_par_scores(df,hrs_d,scores_d,progress=progress)
    print(f"calculate_ddtricks_par_scores: time:{time.time()-t} seconds")

    t = time.time()
    sd_dfs_d, sd_probs_df = calculate_sd_probs(df,hrs_d,sd_productions,progress)
    print(f"calculate_sd_probs: time:{time.time()-t} seconds")

    t = time.time()
    sd_ev_df = calculate_sd_expected_values(df,hrs_d,scores_df)
    print(f"calculate_sd_expected_values: time:{time.time()-t} seconds")

    t = time.time()
    best_contracts_df = create_best_contracts(sd_ev_df)
    print(f"create_best_contracts: time:{time.time()-t} seconds")
    df = pl.concat([df,DDTricks_df,par_df,dd_score_df,sd_probs_df,sd_ev_df,best_contracts_df],how='horizontal')
    
    t = time.time()
    if 'Player_Name_N' in df.columns:
        df = df.rename({'Player_Name_N':'N','Player_Name_E':'E','Player_Name_S':'S','Player_Name_W':'W'}) # todo: is this really better?
        print(f"rename_players: time:{time.time()-t} seconds")

    # cleanup contract column
    t = time.time()
    df = df.with_columns(
        pl.Series('Contract',convert_contract_to_contract(df),pl.String,strict=False), # can have nulls or Strings
    )
    print(f"convert_contract_to_contract: time:{time.time()-t} seconds")

    t = time.time()
    df = df.with_columns(
        pl.Series('Declarer_Direction',convert_contract_to_declarer(df),pl.String,strict=False), # can have nulls or Strings
        pl.Series('BidLvl',convert_contract_to_level(df),pl.UInt8,strict=False), # can have nulls or Strings
        pl.Series('BidSuit',convert_contract_to_strain(df),pl.String,strict=False), # can have nulls or Strings
        pl.Series('Dbl',convert_contract_to_dbl(df),pl.String,strict=False), # can have nulls or Strings
    )
    print(f"convert_contract_to_contract_parts: time:{time.time()-t} seconds")

    # create a column classifying contract type: Pass, Partial, Game, SSlam, GSlam
    t = time.time()
    df = df.with_columns(
        pl.when(pl.col('Contract').eq('PASS')).then(pl.lit("Pass"))
        .when(pl.col('BidLvl').eq(5) & pl.col('BidSuit').is_in(['C', 'D'])).then(pl.lit("Game"))
        .when(pl.col('BidLvl').is_in([4,5]) & pl.col('BidSuit').is_in(['H', 'S'])).then(pl.lit("Game"))
        .when(pl.col('BidLvl').is_in([3,4,5]) & pl.col('BidSuit').eq('N')).then(pl.lit("Game"))
        .when(pl.col('BidLvl').eq(6)).then(pl.lit("SSlam"))
        .when(pl.col('BidLvl').eq(7)).then(pl.lit("GSlam"))
        .otherwise(pl.lit("Partial"))
        .alias('ContractType'),
    )
    print(f"create_contract_types: time:{time.time()-t} seconds")

    t = time.time()
    # todo: replace dicts with generic direction conversion?
    # ACBL assigns Declarer_Direction of 'N' if PASS. We've changed it to None above.
    declarer_to_LHO_d = {None:None,'N':'E','E':'S','S':'W','W':'N'}
    declarer_to_dummy_d = {None:None,'N':'S','E':'W','S':'N','W':'E'}
    declarer_to_RHO_d = {None:None,'N':'W','E':'N','S':'E','W':'S'}
    df = df.with_columns(
        pl.col('Declarer_Direction').replace_strict(declarer_to_LHO_d).alias('LHO_Direction'),
        pl.col('Declarer_Direction').replace_strict(declarer_to_dummy_d).alias('Dummy_Direction'),
        pl.col('Declarer_Direction').replace_strict(declarer_to_RHO_d).alias('RHO_Direction'),
    )
    print(f"convert_contract_to_directions: time:{time.time()-t} seconds")

    t = time.time()
    df = df.with_columns(
        pl.Series('Declarer_Name',convert_declarer_to_DeclarerName(df),pl.String,strict=False), # can have nulls or Strings
    )
    print(f"convert_declarer_to_DeclarerName: time:{time.time()-t} seconds")

    t = time.time()
    df = df.with_columns(
        pl.Series('Declarer_ID',convert_declarer_to_DeclarerID(df),pl.String,strict=False), # can have nulls. endplay has no numeric ids
    )
    print(f"convert_declarer_to_DeclarerID: time:{time.time()-t} seconds")

    t = time.time()
    if 'Result' not in df.columns:
        assert 'Contract' in df.columns, 'Contract column is required to create Result column.' # todo: implement creating of Result from Tricks column.
        df = df.with_columns(
            pl.Series('Result',convert_contract_to_result(df),pl.Int8,strict=False), # can have nulls or Int8
        )
        print(f"convert_contract_to_result: time:{time.time()-t} seconds")

    t = time.time()
    if 'Tricks' not in df.columns:
        df = df.with_columns(
            pl.Series('Tricks',convert_contract_to_tricks(df),pl.UInt8,strict=False), # can have nulls or UInt8
        )
        print(f"convert_contract_to_tricks: time:{time.time()-t} seconds")

    t = time.time()
    if 'DDTricks' not in df.columns:
        df = df.with_columns(
            pl.Series('DDTricks',convert_contract_to_DDTricks(df),pl.UInt8,strict=False), # can have nulls or UInt8
        )
        df = df.with_columns(
            pl.Series('DDTricks_Dummy',convert_contract_to_DDTricks_Dummy(df),pl.UInt8,strict=False), # can have nulls or UInt8
        )
        print(f"convert_contract_to_DDTricks: time:{time.time()-t} seconds")

    t = time.time()
    if 'DDScore_NS' not in df.columns:
        df = convert_contract_to_DDScore_Ref(df)
        print(f"convert_contract_to_DDScore_Ref: time:{time.time()-t} seconds")

    t = time.time()
    if 'Score_NS' not in df.columns:
        df = df.with_columns(
            pl.col('Score').alias('Score_NS'),
            pl.col('Score').neg().alias('Score_EW')
        )
        print(f"convert_score_to_score: time:{time.time()-t} seconds")

    
    # create EV Max and MaxCol with consideration to vulnerability
    t = time.time()
    pair_directions = ['NS', 'EW']
    vul_conditions = {
        'NS': pl.col('Vul').is_in(['N_S', 'Both']),
        'EW': pl.col('Vul').is_in(['E_W', 'Both'])
    }

    # Using already created EV columns (Vul and not vul), creates new columns of Max values and columns of the column names of the max value.
    # # Define the combinations
    # pair_directions = ['NS', 'EW']
    # declarer_directions = 'NESW'
    # strains = 'SHDCN'
    # vulnerabilities = ['NV', 'V']
    max_expressions = []

    for pd in ['NS','EW']:
        # Basic EV Max columns
        max_expressions.extend([
            pl.when(vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_V_Max'))
              .otherwise(pl.col(f'EV_{pd}_NV_Max'))
              .alias(f'EV_{pd}_Max'),

            pl.when(vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_V_MaxCol'))
              .otherwise(pl.col(f'EV_{pd}_NV_MaxCol'))
              .alias(f'EV_{pd}_MaxCol')
        ])

        # For each declarer direction
        for dd in pd: #'NESW':
            max_expressions.extend([
                pl.when(vul_conditions[pd])
                  .then(pl.col(f'EV_{pd}_{dd}_V_Max'))
                  .otherwise(pl.col(f'EV_{pd}_{dd}_NV_Max'))
                  .alias(f'EV_{pd}_{dd}_Max'),

                pl.when(vul_conditions[pd])
                  .then(pl.col(f'EV_{pd}_{dd}_V_MaxCol'))
                  .otherwise(pl.col(f'EV_{pd}_{dd}_NV_MaxCol'))
                  .alias(f'EV_{pd}_{dd}_MaxCol')
            ])

            # For each strain
            for s in 'SHDCN':
                max_expressions.extend([
                    pl.when(vul_conditions[pd])
                      .then(pl.col(f'EV_{pd}_{dd}_{s}_V_Max'))
                      .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_NV_Max'))
                      .alias(f'EV_{pd}_{dd}_{s}_Max'),

                    pl.when(vul_conditions[pd])
                      .then(pl.col(f'EV_{pd}_{dd}_{s}_V_MaxCol'))
                      .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_NV_MaxCol'))
                      .alias(f'EV_{pd}_{dd}_{s}_MaxCol')
                ])
                # For each level
                for l in range(1,8):
                    max_expressions.extend([
                        pl.when(vul_conditions[pd])
                        .then(pl.col(f'EV_{pd}_{dd}_{s}_{l}_V'))
                        .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_{l}_NV'))
                        .alias(f'EV_{pd}_{dd}_{s}_{l}'),
                    ])

    # Apply all expressions at once
    df = df.with_columns(max_expressions)
    df = df.with_columns(
        pl.max_horizontal('EV_NS_Max','EV_EW_Max').alias('EV_Max'),
        pl.max_horizontal('EV_NS_MaxCol','EV_EW_MaxCol').alias('EV_MaxCol')
    )
    print(f"create EV Max and MaxCol with consideration to vulnerability: time:{time.time()-t} seconds")

    # todo: aren't there other diffs that should be create here?
    t = time.time()
    df = df.with_columns(
        pl.Series('ParScore_Diff_NS',(df['Score_NS']-df['ParScore_NS']),pl.Int16),
        pl.Series('ParScore_Diff_EW',(df['Score_EW']-df['ParScore_EW']),pl.Int16),
        # needs to have .cast(pl.Int8) because left and right are both UInt8 which goofs up the subtraction.
        pl.Series('DDTricks_Diff',(df['Tricks'].cast(pl.Int8)-df['DDTricks'].cast(pl.Int8)),pl.Int8,strict=False), # can have nulls or Int8
        pl.Series('EV_MaxScore_Diff_NS',df['Score_NS'] - df['EV_NS_Max'],pl.Float32),
        pl.Series('EV_MaxScore_Diff_EW',-df['Score_NS'] - df['EV_EW_Max'],pl.Float32)
    )
    print(f"create ParScore, DDTricks, EV_MaxScore diffs: time:{time.time()-t} seconds")

    t = time.time()
    df = df.with_columns(
        pl.Series('ParScore_Diff_EW',-df['ParScore_Diff_NS'],pl.Int16), # used for open-closed room comparisons
    )
    print(f"create ParScore_Diff_EW: time:{time.time()-t} seconds")

    return df


# example of working ranking code. but column must contain all scores.
# scores = pl.col('Score_NS')
# df = df.with_columns(
#     scores.rank(method='average', descending=False).sub(1).over(['session_id', 'Board']).alias('ParScore_MP_NS'),
#     scores.rank(method='average', descending=True).sub(1).over(['session_id', 'Board']).alias('ParScore_MP_EW')
# )


def calculate_matchpoint_scores_ns(df,score_columns):
    
    # Process each row
    mp_columns = defaultdict(list)
    for r in df.iter_rows(named=True):
        scores_list = r['Expanded_Scores_List'] # todo: make 'Expanded_Scores_List' sorted and with a 'Score_NS' removed?
        if scores_list is None:
            # todo: kludge: apparently Expanded_Scores_List can be null if director's adjustment.
            # matchpoints can't be computed because there's no list of scores. we only have scores at player's table, not all tables.
            # The fix is to download all table's results to replace a null Expanded_Scores_List.
            print(f"Expanded_Scores_List is null: Score_NS:{r['Score_NS']} MP_NS:{r['MP_NS']}. Skipping.")
            for col in score_columns:
                mp_columns['MP_'+col].append(0.5)
            continue
        scores_list.remove(r['Score_NS'])
        
        for col in score_columns:
            # Calculate rank for each DD score
            rank = 0.0
            new_score = r[col]
            if scores_list:
                for score in scores_list:
                    if new_score > score:
                        rank += 1.0
                    elif new_score == score:
                        rank += 0.5
                    
            mp_columns['MP_'+col].append(rank)
    
    # Add all new columns at once
    return df.hstack(pl.DataFrame(mp_columns))


def PerformMatchPointAndPercentAugmentations(df):

    # todo: not right. some overlap with code in ffbridgelib.convert_ffdf_to_mldf()

    t = time.time()
    if 'MP_Top' not in df.columns:
        # calculate top score (number of board scores - 1)
        df = df.with_columns(
            pl.col('Score').count().over(['session_id','PBN','Board']).sub(1).alias('MP_Top'),
        )
        print(f"create MP_Top: time:{time.time()-t} seconds")

    # todo: check if there's overlap with code in ffbridgelib.convert_ffdf_to_mldf()?
    t = time.time()
    if 'MP_NS' not in df.columns:
        # Calculate matchpoints
        df = df.with_columns([
                # calculate top score which is number of scores in each group - 1
                # calculate matchpoints using rank() and average method
                # assumes 'Score' column contains all scores for the session. if not, _to_mldf() needs to be updated.
                pl.col('Score_NS').rank(method='average', descending=False).sub(1).over(['session_id', 'PBN', 'Board']).alias('MP_NS'),
                pl.col('Score_EW').rank(method='average', descending=False).sub(1).over(['session_id', 'PBN', 'Board']).alias('MP_EW'),
        ])
        print(f"calculate matchpoints MP_(NS|EW): time:{time.time()-t} seconds")

    t = time.time()
    if 'Pct_NS' not in df.columns:
        # Calculate percentages using (n-1) as the top
        df = df.with_columns([
            (pl.col('MP_NS') / pl.col('MP_Top')).alias('Pct_NS'),
            (pl.col('MP_EW') / pl.col('MP_Top')).alias('Pct_EW')
        ])
        print(f"calculate matchpoints percentages MP_(NS|EW): time:{time.time()-t} seconds")

    t = time.time()
    if 'Declarer_Pct' not in df.columns:
        df = df.with_columns(
            pl.when(pl.col('Declarer_Direction').is_in(['N','S']))
            .then('Pct_NS')
            .otherwise('Pct_EW')
            .alias('Declarer_Pct'),
        )
        print(f"create Declarer_Pct: time:{time.time()-t} seconds")

    discrete_score_columns = ['DDScore_NS','ParScore_NS','EV_NS_Max'] # todo: EV needs {Vul} replacement. Use NV for now.'
    dd_score_columns = [f'DDScore_{l}{s}_{d}' for d in 'NESW' for s in 'SHDCN' for l in range(1,8)]
    # EV_{pd}_{dd}_{s}_[1-7]_{v}
    ev_score_columns = [f'EV_{pd}_{d}_{s}_{l}' for pd in ['NS','EW'] for d in pd for s in 'SHDCN' for l in range(1,8)]
    all_score_columns = discrete_score_columns+dd_score_columns+ev_score_columns
    if 'Expanded_Scores_List' in df.columns: # ffbridge only
        print('Calculate matchpoints for existing Expanded_Scores_List column.')
        df = calculate_matchpoint_scores_ns(df,all_score_columns)
    else:
        print('Calculate matchpoints for session, PBN, and Board.')
        for col in all_score_columns:
            if 'MP_'+col not in df.columns:
                # Calculate matchpoints
                df = df.with_columns([
                        # calculate top score which is number of scores in each group - 1
                        # calculate matchpoints using rank() and average method
                        # assumes 'Score' column contains all scores for the session. if not, _to_mldf() needs to be updated.
                        pl.col('Score_NS').rank(method='average', descending=False).sub(1).over(['session_id', 'PBN', 'Board']).alias('MP_'+col)
                ])
    print(f"calculate matchpoints all_score_columns: time:{time.time()-t} seconds")

    t = time.time()
    for col_ns in discrete_score_columns:
        col_ew = col_ns.replace('NS','EW')
        df = df.with_columns(
            (pl.col('MP_Top')-pl.col(f'MP_{col_ns}')).alias(f'MP_{col_ew}')
        )
        df = df.with_columns(
            (pl.col(f'MP_{col_ns}')/pl.col('MP_Top')).alias(col_ns.replace('_NS','_Pct_NS')),
            (pl.col(f'MP_{col_ew}')/pl.col('MP_Top')).alias(col_ew.replace('_EW','_Pct_EW')),
        )

    df = df.with_columns([
        (1-pl.col('ParScore_Pct_NS')).alias('ParScore_Pct_EW'),
    ])
    df = df.with_columns([
        pl.max_horizontal(f'^MP_DDScore_[1-7][SHDCN]_[NS]$').alias(f'MP_DDScore_NS_Max'),
    ])
    df = df.with_columns([
        pl.max_horizontal(f'^MP_DDScore_[1-7][SHDCN]_[EW]$').alias(f'MP_DDScore_EW_Max'),
    ])
    df = df.with_columns([
        pl.max_horizontal(f'^MP_EV_NS_[NS]_[SHDCN]_[1-7]$').alias(f'MP_EV_NS_Max'),
    ])
    df = df.with_columns([
        pl.max_horizontal(f'^MP_EV_EW_[EW]_[SHDCN]_[1-7]$').alias(f'MP_EV_EW_Max'),
    ])

    df = df.with_columns([
        (pl.col('MP_DDScore_NS_Max')/pl.col('MP_Top')).alias('DDScore_Pct_NS_Max'),
        (pl.col('MP_DDScore_EW_Max')/pl.col('MP_Top')).alias('DDScore_Pct_EW_Max'),
        (pl.col('MP_EV_NS_Max')/pl.col('MP_Top')).alias('EV_Pct_NS_Max'),
        (pl.col('MP_EV_EW_Max')/pl.col('MP_Top')).alias('EV_Pct_EW_Max'),

        pl.col('DDScore_Pct_NS').alias('DDPct_NS'),
        pl.col('DDScore_Pct_EW').alias('DDPct_EW'),
        pl.col('MP_NS').alias('Matchpoints_NS'),
        pl.col('MP_EW').alias('Matchpoints_EW'),
        pl.col('MP_EV_NS_Max').alias('SDMPs_Max_NS'),
        pl.col('MP_Top').sub(pl.col('MP_EV_NS_Max')).alias('SDMPs_Max_EW'),
        #pl.col('MP_EV_NS_Max').alias('MP_EV_NS_Max'), # same
        #pl.col('MP_EV_EW_Max').alias('MP_EV_EW_Max'), # same
        # SDScore_Max_NS
        pl.col('EV_Pct_NS_Max').alias('SDPct_NS'),
        pl.col('EV_Pct_EW_Max').alias('SDPct_EW'),
        pl.col('EV_Pct_NS_Max').alias('SDPct_Max_NS'),
        pl.col('EV_Pct_EW_Max').alias('SDPct_Max_EW'),
        (pl.col('EV_Pct_NS_Max')-pl.col('Pct_NS')).alias('SDPct_Max_Diff_NS'),
        (pl.col('EV_Pct_EW_Max')-pl.col('Pct_EW')).alias('SDPct_Max_Diff_EW'),
        (pl.col('ParScore_Pct_NS')-pl.col('Pct_NS')).alias('SDParScore_Pct_Diff_NS'),
        (pl.col('ParScore_Pct_EW')-pl.col('Pct_EW')).alias('SDParScore_Pct_Diff_EW'),
        (pl.col('ParScore_Pct_NS')-pl.col('Pct_NS')).alias('SDParScore_Pct_Max_Diff_NS'),
        (pl.col('ParScore_Pct_EW')-pl.col('Pct_EW')).alias('SDParScore_Pct_Max_Diff_EW'),
        ])

    # test sql query: FROM self SELECT Board, Contract, Score, Score_NS, Score_EW, ParScore_NS, Expanded_Scores_List, MP_NS, MP_EW, MP_ParScore_NS, MP_ParScore_EW, ParScore_Pct_NS, ParScore_Pct_EW, DDScore_3N_N, MP_DDScore_3N_N, MP_DDScore_NS, MP_DDScore_EW, MP_EV_NS, MP_EV_EW, DDScore_Pct_NS, DDScore_Pct_EW, EV_NS_NV_Max, EV_EW_NV_MaxMP_EV_NS, MP_EV_EW, EV_Pct_NS, EV_Pct_EW, EV_NS_N_H_4_NV, EV_EW_E_H_4_NV
    # test sql query: SELECT Board, Contract, Score, Score_NS, Score_EW, ParScore_NS, ParScore_EW, SDScore, SDScore_NS, SDScore_EW
    print(f"Time to rank expanded scores: {time.time()-t} seconds")

    return df


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


# todo: refactor. Most of PerformResultAugmentations should be in hand record augmentation, not result augmentation.
def PerformResultAugmentations(df,hrs_d):

    # create column of Hands expressed in binary.
    # if 'Hands_Bin' in df.columns:
    #     print('Hands_Bin already exists. skipping...')
    # else:
    #     # takes 18m
    #     t = time.time()
    #     hbs_l = [mlBridgeLib.HandsToBin(hands) for hands in df['Hands']]
    #     df = df.with_columns(pl.Series('Hands_Bin',hbs_l,pl.Object))
    #     del hbs_l
    #     print(f"Time to create Hands_Bin: {time.time()-t} seconds")
    # print(df[['Hands','Hands_Bin']])

    # Create one hot encoding, length 52, for each direction's hand.
    # todo: one hot encode each direction's hand? Leaving as binary string for now.
    # if 'HB_N' in df.columns:
    #     print('HB_N already exists. skipping...')
    # else:
    #     # takes 30s
    #     t = time.time()
    #     hands_bin_d = OHE_Hands(df['Hands_Bin'])
    #     hands_bin_df = pl.DataFrame(hands_bin_d)
    #     df = pl.concat([df,hands_bin_df],how='horizontal')
    #     del hands_bin_df,hands_bin_d
    #     print(f"Time to create HB_N: {time.time()-t} seconds")
    # print(df[['Hands','HB_N','HB_E','HB_S','HB_W']])

    if 'C_NSA' in df.columns:
        print('C_NSA already exists. skipping...')
    else:
        # takes 1m
        t = time.time()
        df = DealToCards(df)
        print(f"Time to create C_NSA: {time.time()-t} seconds")

    # Compute HCPs from Hands. Validate against any existing HCP column.
    if 'HCP_N_C' in df.columns:
        print('HCP_N_C already exists. skipping...')
    else:
        # takes 16m
        t = time.time()
        df = CardsToHCP(df)
        print(f"Time to create HCP: {time.time()-t} seconds")

    # Compute quick tricks from Hands
    if 'QT_N_C' in df.columns:
        print('QT_N_C already exists. skipping...')
    else:
        # takes 10m
        t = time.time()
        df = CardsToQuickTricks(df)
        print(f"Time to create QT: {time.time()-t} seconds")

    # Compute suit lengths from Hands
    if 'SL_N_C' in df.columns:
        print('SL_N_C already exists. skipping...')
    else:
        # takes 9m
        t = time.time()
        # Create a list of new column expressions
        sl_nesw_columns = [
            pl.col(f"Suit_{direction}_{suit}").str.len_chars().alias(f"SL_{direction}_{suit}")
            for direction in "NESW"
            for suit in "SHDC"
        ]
        df = df.with_columns(sl_nesw_columns)
        print(f"Time to create SL_[NESW]_[SHDC]: {time.time()-t} seconds")

    if 'SL_NS_C' in df.columns:
        print('SL_NS_C already exists. skipping...')
    else:
        # takes 9m
        t = time.time()
        sl_ns_ew_columns = [
            pl.sum_horizontal(f"SL_{pair[0]}_{suit}",f"SL_{pair[1]}_{suit}").alias(f"SL_{pair}_{suit}")
            for pair in ['NS','EW']
            for suit in "SHDC"
        ]
        df = df.with_columns(sl_ns_ew_columns)
        print(f"Time to create SL_(NS|EW)_[SHDC]: {time.time()-t} seconds")

    if 'SL_N_CDHS' in df.columns:
        print('SL_N_CDHS already exists. skipping...')
        assert 'SL_N_CDHS_SJ' in df.columns and 'SL_N_ML' in df.columns and 'SL_N_ML_SJ' in df.columns and 'SL_N_ML_I' in df.columns and 'SL_N_ML_I_SJ' in df.columns
    else:
        # takes 17m15s-22m for 10k rows
        t = time.time()
        for d in 'NESW':
            cdhs_l = df[[f"SL_{d}_{s}" for s in 'CDHS']].rows() # CDHS suit lengths
            ml_li_l = [sorted([(l,i) for i,l in enumerate(r)],reverse=True) for r in df[[f"SL_{d}_{s}" for s in 'CDHS']].rows()] # (length,index) ex: (4,3),(4,0),(3,1),(2,2)
            ml_l = [[t2[0]  for t2 in t4] for t4 in ml_li_l] # most-to-least lengths
            ml_i_l = [[t2[1]  for t2 in t4] for t4 in ml_li_l] # column indices of most-to-least lengths
            df = df.with_columns(
                pl.Series(f'SL_{d}_CDHS',cdhs_l,pl.Array(pl.UInt8,shape=(4,))), # array of CDHS suit lengths
                pl.Series(f'SL_{d}_CDHS_SJ',['-'.join(map(str,r)) for r in cdhs_l],pl.String), # CDHS suit lengths stringized and joined
                pl.Series(f"SL_{d}_ML",ml_l,pl.Array(pl.UInt8,shape=(4,))), # most-to-least suit lengths
                pl.Series(f"SL_{d}_ML_SJ",['-'.join(map(str,r)) for r in ml_l],pl.String), # most-to-least suit lengths stringized and joined
                pl.Series(f"SL_{d}_ML_I",ml_i_l,pl.Array(pl.UInt8,shape=(4,))), # column indices of most-to-least
                pl.Series(f"SL_{d}_ML_I_SJ",['-'.join(map(str,r)) for r in ml_i_l],pl.String), # column indices of most-to-least stringized and joined
            )
        print(f"Time to create SL_[NESW]_CDHS.* and SL_[NESW]_ML.*: {time.time()-t} seconds")

    # Calculate distribution points using 3-2-1 system.
    if 'DP_N_C' in df.columns:
        print('DP_N_C already exists. skipping...')
    else:
        t = time.time()
        dp_columns = [
            pl.when(pl.col(f"SL_{direction}_{suit}") == 0).then(3)
            .when(pl.col(f"SL_{direction}_{suit}") == 1).then(2)
            .when(pl.col(f"SL_{direction}_{suit}") == 2).then(1)
            .otherwise(0)
            .alias(f"DP_{direction}_{suit}")
            for direction in "NESW"
            for suit in "SHDC"
        ]
        df = df.with_columns(dp_columns)
        df = df.with_columns(
            (pl.col('DP_N_S')+pl.col('DP_N_H')+pl.col('DP_N_D')+pl.col('DP_N_C')).alias('DP_N'),
            (pl.col('DP_S_S')+pl.col('DP_S_H')+pl.col('DP_S_D')+pl.col('DP_S_C')).alias('DP_S'),
            (pl.col('DP_E_S')+pl.col('DP_E_H')+pl.col('DP_E_D')+pl.col('DP_E_C')).alias('DP_E'),
            (pl.col('DP_W_S')+pl.col('DP_W_H')+pl.col('DP_W_D')+pl.col('DP_W_C')).alias('DP_W'),
        )
        df = df.with_columns(
            (pl.col('DP_N')+pl.col('DP_S')).alias('DP_NS'),
            (pl.col('DP_E')+pl.col('DP_W')).alias('DP_EW'),
        )
        print(f"Time to create DP_[NESW]_[SHDC] DP_[NESW] DP_(NS|EW): {time.time()-t} seconds")

    # Calculate total points from HCP and DP.
    if 'Total_Points_N_C' in df.columns:
        print('Total_Points_N_C already exists. skipping...')
    else:
        print(f"Todo: Don't forget to adjust Total_Points for singleton king and doubleton queen.")
        t = time.time()
        df = df.with_columns(
            (
                # adjust points for singleton king and doubleton queen. pl.when(min(hcp,dp)) or just subtract 1?
                (pl.col(f'HCP_{direction}_{suit}')+pl.col(f'DP_{direction}_{suit}')).alias(f'Total_Points_{direction}_{suit}')
                for direction in 'NESW'
                for suit in 'SHDC'
            )
        )
        df = df.with_columns(
            (
                (pl.col(f'Total_Points_{direction}_S')+pl.col(f'Total_Points_{direction}_H')+pl.col(f'Total_Points_{direction}_D')+pl.col(f'Total_Points_{direction}_C')).alias(f'Total_Points_{direction}')
                for direction in 'NESW'
            )
        )
        df = df.with_columns(
            (pl.col('Total_Points_N')+pl.col('Total_Points_S')).alias('Total_Points_NS'),
            (pl.col('Total_Points_E')+pl.col('Total_Points_W')).alias('Total_Points_EW'),
        )
        print(f"Time to create Total_Points_[NESW]_[SHDC] Total_Points_[NESW] Total_Points_(NS|EW): {time.time()-t} seconds")

    if 'SL_Max_NS' in df.columns:
        print('SL_Max_NS already exists. skipping...')
    else:
        # takes 15s
        t = time.time()
        sl_cols = [('_'.join(['SL_Max',d]),['_'.join(['SL',d,s]) for s in mlBridgeLib.SHDC]) for d in mlBridgeLib.NS_EW]
        # Create columns containing column names of the NS,EW longest suit.
        for d in sl_cols:
            df = df.with_columns(pl.Series(d[0],[d[1][l.index(max(l))] for l in df[d[1]].rows()])) #.cast(pl.Categorical)) #.alias(d[0])) # defaults to object so need string or category
        #for d_ns,d_ew in df[['SL_Max_NS','SL_Max_EW']].rows():
        #    df = df.with_columns(pl.max_horizontal(f'DD_{d_ns[-4]}_{d_ns[-1]}',f'DD_{d_ew[-4]}_{d_ew[-1]}'),d_ns[-1]).alias(f'DD_Max_NS_{d_ns[-1]}')
        print(f"Time to create SL_Max_(NS|EW): {time.time()-t} seconds")

    assert 'ParScore_NS' in df.columns
    # if 'ParScore_NS' in df.columns:
    #     print('ParScore_NS already exists. skipping...')
    # else:
    #     # takes 15s
    #     t = time.time()
    #     Pars_l = [hrs_d[pbn]['Par'][(d,v)] for pbn,d,v in df[('PBN','Dealer','Vul')].rows()] # 'Par' is hrs_d's legacy name for ParScore_NS
    #     df = df.with_columns(pl.Series('ParScore_NS',Pars_l,pl.Object)) # todo: specify correct dtype instead of object
    #     df = df.with_columns(pl.Series('ParScore_EW',-df['ParScore_NS'],pl.Object)) # todo: specify correct dtype instead of object
    #     print(f"Time to create ParScore_NS: {time.time()-t} seconds")

    if 'LoTT' in df.columns:
        print('LoTT already exists. skipping...')
    else:
        # takes 1m30s
        t = time.time()
        df = calculate_LoTT(df)
        print(f"Time to create LoTT: {time.time()-t} seconds")

    # Create column of contract types by partnership by suit. e.g. CT_NS_C.
    # rename to DD_CT_[NESW]_[SHDC]
    if 'CT_N_C' in df.columns:
        print('CT_N_C already exists. skipping...')
    else:
        t = time.time()
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
        df = df.with_columns(ct_columns)
        print(f"Time to create CT_(NS|EW)_[SHDCN]: {time.time()-t} seconds")

    # Create columns of contract type booleans by direction by suit by contract. e.g. CT_N_C_Game
    if 'CT_N_C_Game' in df.columns:
        print('CT_N_C_Game already exists. skipping...')
    else:
        # takes 5s
        t = time.time()
        ct_boolean_columns = [
            pl.col(f"CT_{direction}_{strain}").eq(pl.lit(contract))
            .alias(f"CT_{direction}_{strain}_{contract}")
            for direction in "NESW"
            for strain in "SHDCN"
            for contract in ["Pass","Game","SSlam","GSlam","Partial"]
        ]
        df = df.with_columns(ct_boolean_columns)
        print(f"Time to create CT_(NS|EW)_[SHDCN]_(Pass|Game|SSlam|GSlam|Partial): {time.time()-t} seconds")


    # Create columns of dealer by board number. This works only if vulnerability follows usual board numbering. Not so for board data.
    # todo: is this in the right place?
    if 'Dealer' in df.columns:
        print('Dealer already exists. skipping...')
    else:
        # takes 5s
        t = time.time()

        def BoardNumberToDealer(bn):
            return 'NESW'[(bn-1) & 3]

        df = df.with_columns(pl.col('board_boardNumber').map_elements(BoardNumberToDealer,return_dtype=pl.String).alias('Dealer'))
        print(f"Time to create Dealer: {time.time()-t} seconds")


    # Create columns of vulnerability by board number.
    # todo: is this in the right place?
    if 'iVul' in df.columns:
        print('iVul already exists. skipping...')
    else:
        # takes 5s
        t = time.time()

        if 'Vul' in df.columns:

            def VulToiVul(vul):
                return ['None','N_S','E_W','Both'].index(vul)

            df = df.with_columns(
                pl.col('Vul')
                    .map_elements(VulToiVul,return_dtype=pl.UInt8)
                .alias('iVul')
            )

        else:

            def BoardNumberToVul(bn):
                bn -= 1
                return range(bn//4, bn//4+4)[bn & 3] & 3

            df = df.with_columns(
                pl.col('Board')
                    .map_elements(BoardNumberToVul,return_dtype=pl.UInt8)
                .alias('iVul')
            )
        
        print(f"Time to create iVul_(NS|EW): {time.time()-t} seconds")


    # Create columns of vulnerability from iVul. iVul already exists.
    # todo: is this in the right place?
    if 'Vul' in df.columns:
        print('Vul already exists. skipping...')
    else:
        # takes 5s
        t = time.time()
        
        def iVulToVul(ivul):
            return ['None','N_S','E_W','Both'][ivul]

        df = df.with_columns(
            pl.col('iVul')
                .map_elements(iVulToVul,return_dtype=pl.String)
            .alias('Vul')
        )
        print(f"Time to create Vul_(NS|EW): {time.time()-t} seconds")


    # Create columns of vulnerability by partnership.
    if 'Vul_NS' in df.columns:
        print('Vul_NS already exists. skipping...')
    else:
        # takes 5s
        t = time.time()
        df = df.with_columns(
            pl.Series('Vul_NS',df['Vul'].is_in(['N_S','Both']),pl.Boolean),
            pl.Series('Vul_EW',df['Vul'].is_in(['E_W','Both']),pl.Boolean)
        )
        print(f"Time to create Vul_(NS|EW): {time.time()-t} seconds")

    t = time.time()
    # Define the criteria for each series type
    suit_quality_criteria = {
        "Biddable": lambda sl, hcp: sl.ge(5) | (sl.eq(4) & hcp.ge(3)),
        "Rebiddable": lambda sl, hcp: sl.ge(6) | (sl.eq(5) & hcp.ge(3)),
        "Twice_Rebiddable": lambda sl, hcp: sl.ge(7) | (sl.eq(6) & hcp.ge(3)),
        "Strong_Rebiddable": lambda sl, hcp: sl.ge(6) & hcp.ge(9),
        "Solid": lambda sl, hcp: hcp.ge(9),  # todo: 6 card requires ten
    }

    # Stopper criteria from worst to best
    stopper_criteria = {
        "At_Best_Partial_Stop_In": lambda sl, hcp: (sl + hcp).lt(4),  # todo: seems wrong
        "Partial_Stop_In": lambda sl, hcp: (sl + hcp).ge(4),
        "Likely_Stop_In": lambda sl, hcp: (sl + hcp).ge(5),
        "Stop_In": lambda sl, hcp: hcp.ge(4) | (sl + hcp).ge(6),
        "At_Best_Stop_In": lambda sl, hcp: (sl + hcp).ge(7),
        "Two_Stops_In": lambda sl, hcp: (sl + hcp).ge(8),
    }

    # Create all series expressions
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

    # Apply all expressions at once
    df = df.with_columns(series_expressions)

    df = df.with_columns(
        pl.lit(False).alias(f"Forcing_One_Round"), # todo
        pl.lit(False).alias(f"Opponents_Cannot_Play_Undoubled_Below_2N"), # todo
        pl.lit(False).alias(f"Forcing_To_2N"), # todo
        pl.lit(False).alias(f"Forcing_To_3N"), # todo
    )

    # Create balanced hand indicators for each direction
    # A hand is considered balanced if it has one of these distributions:
    # - 4-3-3-3 
    # - 4-4-3-2
    # - 5-3-3-2 (only if the 5-card suit is clubs or diamonds)
    # - 5-4-2-2 (only if the 5-card suit is clubs or diamonds)
    df = df.with_columns(
        pl.Series(f"Balanced_{direction}",df[f"SL_{direction}_ML_SJ"].is_in(['4-3-3-3','4-4-3-2'])
            | (df[f"SL_{direction}_ML_SJ"].is_in(['5-3-3-2','5-4-2-2']) & (df[f"SL_{direction}_C"].eq(5) | df[f"SL_{direction}_D"].eq(5))),pl.Boolean)
        for direction in 'NESW'
    )
    print(f"Time to create misc: {time.time()-t} seconds")

    return df

# additional augmentations for ACBL hand records
def AugmentACBLHandRecords(df,hrs_d):

    df = perform_hand_augmentations(df,hrs_d)

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

    df = df.with_columns(
        #pl.col('Section').alias('section_name'), # will this be needed for acbl?
        pl.col('N').alias('Player_Name_N'),
        pl.col('S').alias('Player_Name_S'),
        pl.col('E').alias('Player_Name_E'),
        pl.col('W').alias('Player_Name_W'),
        pl.col('Declarer_Name').alias('Name_Declarer'),
        pl.col('Declarer_ID').alias('Number_Declarer'), #  todo: rename to 'Declarer_ID'?
        # todo: rename to 'Declarer_Pair_Direction'
        pl.when(pl.col('Declarer_Direction').is_in(['N','S'])).then(pl.lit('NS')).otherwise(pl.lit('EW')).alias('Pair_Declarer_Direction'),

        # EV legacy renames
        pl.col('EV_MaxCol').alias('SDContract_Max'), # Pair direction invariant.
        pl.col('EV_NS_Max').alias('SDScore_NS'),
        pl.col('EV_EW_Max').alias('SDScore_EW'),
        pl.col('EV_NS_Max').alias('SDScore_Max_NS'),
        pl.col('EV_EW_Max').alias('SDScore_Max_EW'),
        (pl.col('EV_NS_Max')-pl.col('Score_NS')).alias('SDScore_Diff_NS'),
        (pl.col('EV_EW_Max')-pl.col('Score_EW')).alias('SDScore_Diff_EW'),
        (pl.col('EV_NS_Max')-pl.col('Score_NS')).alias('SDScore_Max_Diff_NS'),
        (pl.col('EV_EW_Max')-pl.col('Score_EW')).alias('SDScore_Max_Diff_EW'),
        (pl.col('EV_NS_Max')-pl.col('Pct_NS')).alias('SDPct_Diff_NS'),
        (pl.col('EV_EW_Max')-pl.col('Pct_EW')).alias('SDPct_Diff_EW'),
        #['Probs',pair_direction,declarer_direction,suit,str(i)]
        #([pl.lit(f'Probs_NS_N_S_{t}').alias(f'SDProbs_Taking_{t}') for t in range(14)]), # wrong should be e.g. SDProbs_Taking_0
        pl.col(f'Probs_NS_N_S_0').alias(f'SDProbs_Taking_0'),
        pl.col(f'Probs_NS_N_S_1').alias(f'SDProbs_Taking_1'),
        pl.col(f'Probs_NS_N_S_2').alias(f'SDProbs_Taking_2'),
        pl.col(f'Probs_NS_N_S_3').alias(f'SDProbs_Taking_3'),
        pl.col(f'Probs_NS_N_S_4').alias(f'SDProbs_Taking_4'),
        pl.col(f'Probs_NS_N_S_5').alias(f'SDProbs_Taking_5'),
        pl.col(f'Probs_NS_N_S_6').alias(f'SDProbs_Taking_6'),
        pl.col(f'Probs_NS_N_S_7').alias(f'SDProbs_Taking_7'),
        pl.col(f'Probs_NS_N_S_8').alias(f'SDProbs_Taking_8'),
        pl.col(f'Probs_NS_N_S_9').alias(f'SDProbs_Taking_9'),
        pl.col(f'Probs_NS_N_S_10').alias(f'SDProbs_Taking_10'),
        pl.col(f'Probs_NS_N_S_11').alias(f'SDProbs_Taking_11'),
        pl.col(f'Probs_NS_N_S_12').alias(f'SDProbs_Taking_12'),
        pl.col(f'Probs_NS_N_S_13').alias(f'SDProbs_Taking_13'),
    )
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


def Perform_DD_SD_Augmentations(df):

    df = Perform_Legacy_Renames(df) # todo: update names/SQL to make this unnecessary.
    df = Create_Fake_Predictions(df)

    # todo: temporary(?) aliases until SQL and other df columns are renamed.
    # todo: need to deal with {Vul} replacement by creating row version by selecting NV, V version.

    #print(df.select(pl.col('^EV_.*$')).columns)
    df = df.with_columns(
        # create a column of column names of the SD score of the declarer's contract
        pl.concat_str([
            pl.lit('EV'),
            pl.col('Pair_Declarer_Direction'), # renamed?
            pl.col('Declarer_Direction'),
            pl.col('BidSuit'),
            pl.col('BidLvl').cast(pl.String),
            ], separator='_')
            .alias('Declarer_SDContract'),
        # calculate score in terms of declarer pair direction
        pl.when(pl.col('Pair_Declarer_Direction').eq(pl.lit('NS')))
        .then(pl.col('Score_NS'))
        .otherwise(pl.col('Score_EW'))
        .alias('Score_Declarer'), # todo: rename to 'Declarer_Score'?
        # calculate par score in terms of declarer pair direction
        pl.when(pl.col('Pair_Declarer_Direction').eq(pl.lit('NS')))
        .then(pl.col('ParScore_NS'))
        .otherwise(pl.col('ParScore_EW'))
        .alias('ParScore_Declarer'), # todo: rename to 'Declarer_ParScore'?
        ((pl.col('Pair_Declarer_Direction').eq('NS') & pl.col('Vul_NS')) | (pl.col('Pair_Declarer_Direction').eq('EW') & pl.col('Vul_EW'))).alias('Declarer_Vul'),
    )
    # position columns
    df = df.with_columns(
        pl.col('Declarer_Direction').replace_strict(mlBridgeLib.PlayerDirectionToPairDirection).alias('Pair_Declarer_Direction'),
        pl.col('Declarer_Direction').replace_strict(mlBridgeLib.NextPosition).alias('Direction_OnLead'),
    )
    df = df.with_columns(
        pl.col('Pair_Declarer_Direction').replace_strict(mlBridgeLib.PairDirectionToOpponentPairDirection).alias('Opponent_Pair_Direction'),
        pl.struct(['Pair_Declarer_Direction', 'Score_NS', 'Score_EW']).map_elements(lambda r: None if r['Pair_Declarer_Direction'] is None else r[f'Score_{r["Pair_Declarer_Direction"]}'],return_dtype=pl.Int16).alias('Score_Declarer'),
        pl.col('Direction_OnLead').replace_strict(mlBridgeLib.NextPosition).alias('Direction_Dummy'),
        pl.struct(['Direction_OnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(lambda r: None if r['Direction_OnLead'] is None else r[f'Player_ID_{r["Direction_OnLead"]}'],return_dtype=pl.String).alias('OnLead'),
    )
    df = df.with_columns(
        pl.col('Direction_Dummy').replace_strict(mlBridgeLib.NextPosition).alias('Direction_NotOnLead'),
        pl.struct(['Direction_Dummy', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(lambda r: None if r['Direction_Dummy'] is None else r[f'Player_ID_{r["Direction_Dummy"]}'],return_dtype=pl.String).alias('Dummy'),
        pl.col('Score_Declarer').le(pl.col('ParScore_Declarer')).alias('Defender_ParScore_GE')
    )
    df = df.with_columns(
        pl.struct(['Direction_NotOnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(lambda r: None if r['Direction_NotOnLead'] is None else r[f'Player_ID_{r["Direction_NotOnLead"]}'],return_dtype=pl.String).alias('NotOnLead'),
        pl.struct(['Pair_Declarer_Direction', 'Vul_NS', 'Vul_EW']).map_elements(lambda r: None if r['Pair_Declarer_Direction'] is None else r[f'Vul_{r["Pair_Declarer_Direction"]}'],return_dtype=pl.Boolean).alias('Vul_Declarer'),
        pl.struct(['Pair_Declarer_Direction', 'Pct_NS', 'Pct_EW']).map_elements(lambda r: None if r['Pair_Declarer_Direction'] is None else r[f'Pct_{r["Pair_Declarer_Direction"]}'],return_dtype=pl.Float32).alias('Pct_Declarer'),
        pl.struct(['Pair_Declarer_Direction', 'Pair_Number_NS', 'Pair_Number_EW']).map_elements(lambda r: None if r['Pair_Declarer_Direction'] is None else r[f'Pair_Number_{r["Pair_Declarer_Direction"]}'],return_dtype=pl.UInt32).alias('Pair_Number_Declarer'),
        pl.struct(['Opponent_Pair_Direction', 'Pair_Number_NS', 'Pair_Number_EW']).map_elements(lambda r: None if r['Opponent_Pair_Direction'] is None else r[f'Pair_Number_{r["Opponent_Pair_Direction"]}'],return_dtype=pl.UInt32).alias('Pair_Number_Defender'),
        pl.struct(['Declarer_Direction', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(lambda r: None if r['Declarer_Direction'] is None else r[f'Player_ID_{r["Declarer_Direction"]}'],return_dtype=pl.String).alias('Number_Declarer'),
        pl.struct(['Declarer_Direction', 'Player_Name_N', 'Player_Name_E', 'Player_Name_S', 'Player_Name_W']).map_elements(lambda r: None if r['Declarer_Direction'] is None else r[f'Player_Name_{r["Declarer_Direction"]}'],return_dtype=pl.String).alias('Name_Declarer'),
    )

    # board result columns
    # todo: this func call returns static data but duplicated here. should only be called once. but streamlit misbehaves on globals.
    print(df.filter(pl.col('Result').is_null() | pl.col('Tricks').is_null())['Contract','Declarer_Direction','Declarer_Vul','Vul_Declarer','iVul','Score_NS','BidLvl','Result','Tricks'])
    all_scores_d, scores_d, scores_df = calculate_scores() # todo: this func call returns static data but is duplicated here.
    df = df.with_columns(
        # word to the wise: map_elements() requires every column to be specified in pl.struct() and return_dtype must be compatible.
        # SDScore is the SD score of the declarer's contract.
        # note: cool example of dereferencing a column of column names into a column of values
        pl.struct(['Declarer_SDContract','^EV_(NS|EW)_[NESW]_[SHDCN]_[1-7]$'])
            .map_elements(lambda x: None if x['Declarer_SDContract'] is None else x[x['Declarer_SDContract']],return_dtype=pl.Float32).alias('SDScore'),
        # Computed_Score_Declarer is the computed score of the declarer's contract.
        # note: cool example of calling dict having keys that are tuples
        pl.struct(['BidLvl', 'BidSuit', 'Tricks', 'Vul_Declarer', 'Dbl'])
            .map_elements(lambda x: all_scores_d.get(tuple(x.values()),None),return_dtype=pl.Int16)
            .alias('Computed_Score_Declarer'),

        pl.struct(['Contract', 'Result', 'Score_NS', 'BidLvl', 'BidSuit', 'Dbl','Declarer_Direction', 'Vul_Declarer']).map_elements(
            lambda r: 0 if r['Contract'] == 'PASS' else r['Score_NS'] if r['Result'] is None else mlBridgeLib.score(
                r['BidLvl'] - 1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), 'NESW'.index(r['Declarer_Direction']),
                r['Vul_Declarer'], r['Result'], True),return_dtype=pl.Int16).alias('Computed_Score_Declarer2'),
    )
    # todo: can remove df['Computed_Score_Declarer2'] after assert has proven equality
    # if asserts, may be due to Result or Tricks having nulls.
    assert df['Computed_Score_Declarer'].eq(df['Computed_Score_Declarer2']).all()


    df = df.with_columns(
        (pl.col('Result') > 0).alias('OverTricks'),
        (pl.col('Result') == 0).alias('JustMade'),
        (pl.col('Result') < 0).alias('UnderTricks'),
        # todo: duplicate of 'Computed_Score_Declarer' so can be removed after asserting equal.
       pl.col('Tricks').alias('Tricks_Declarer'),
        (pl.col('Tricks') - pl.col('DDTricks')).alias('Tricks_DD_Diff_Declarer'),
    )

    # Grouped calculation columns using over()
    df = df.with_columns([
        pl.col('Tricks_DD_Diff_Declarer')
        .mean()
        .over('Number_Declarer')
        .alias('Declarer_Rating'),

        pl.col('Defender_ParScore_GE')
        .cast(pl.Float32)
        .mean()
        .over('OnLead')
        .alias('Defender_OnLead_Rating'),

        pl.col('Defender_ParScore_GE')
        .cast(pl.Float32)
        .mean()
        .over('NotOnLead')
        .alias('Defender_NotOnLead_Rating')
    ])

    return df
