# Contains functions for:
# 1. reading endplay compatible files
# 2. creates endplay board classes
# 3. creates an endplay polars df from boards classes
# 4. converts the endplay df to a mlBridge df


import polars as pl
import pickle
from collections import defaultdict

import endplay.parsers.lin as lin


def lin_files_to_boards_dict(lin_files_l,boards_d,bbo_lin_files_cache_file=None):
    load_count = 0
    for i,lin_file in enumerate(lin_files_l):
        if i % 10000 == 0:
            print(f'{i}/{len(lin_files_l)} {load_count=} file:{lin_file}')
        if lin_file in boards_d:
            continue
        with open(lin_file, 'r', encoding='utf-8') as f:
            try:
                boards_d[lin_file] = lin.load(f)
            except Exception as e:
                print(f'error: {i}/{len(lin_files_l)} file:{lin_file} error:{e}')
                continue
        load_count += 1
        if load_count % 1000000 == 0:
            if bbo_lin_files_cache_file is not None:
                with open(bbo_lin_files_cache_file, 'wb') as f:
                    pickle.dump(boards_d,f)
                print(f"Saved {str(bbo_lin_files_cache_file)}: len:{len(boards_d)} size:{bbo_lin_files_cache_file.stat().st_size}")
    return boards_d


def endplay_boards_to_df(boards_d): 
    # Initialize data dictionary as defaultdict(list)
    board_d = defaultdict(list)

    # There's always only one board per lin file.(?). You can have multiple boards by simply concatenating lin files with '\n' in between them.
    for nfiles,(lin_file,boards_in_lin_file) in enumerate(boards_d.items()):
        if nfiles % 10000 == 0:
            print(f'{nfiles}/{len(boards_d)}/{len(boards_in_lin_file)} file:{lin_file}')
        #if nfiles == 100000:
        #    break
        for i,b in enumerate(boards_in_lin_file):
            board_d['board_num'].append(b.board_num)
            board_d['dealer'].append(b.dealer.abbr) # None if passed out
            board_d['vulnerability'].append(b._vul) # None if passed out, weird
            board_d['passout'].append(b._contract.is_passout() if b._contract else None)
            board_d['contract'].append(str(b._contract) if b._contract else None)
            board_d['level'].append(b._contract.level if b._contract else None)
            board_d['denom'].append(b._contract.denom.name if b._contract else None)
            board_d['trump'].append(b.deal.trump.name)
            board_d['penalty'].append(b._contract.penalty.name if b._contract else None)
            board_d['declarer'].append(b._contract.declarer.name if b._contract else None)
            board_d['result'].append(b._contract.result if b._contract else None)
            board_d['score'].append(b._contract.score(b._vul) if b._contract else None)
            board_d['claimed'].append(b.claimed)
            board_d['PBN'].append(str(b.deal.to_pbn())) #.to_pbn())
            board_d['Hand_N'].append(str(b.deal.north)) # str()
            board_d['Hand_E'].append(str(b.deal.east)) # str()
            board_d['Hand_S'].append(str(b.deal.south)) # str()
            board_d['Hand_W'].append(str(b.deal.west)) # str()
            board_d['info'].append(b.info)
            board_d['source_file'].append(str(lin_file)) # todo: change key to be str instead of pathlib.Path?
            bid_type = []
            denom = []
            penalty = []
            level = []
            alertable = []
            announcement = []

            for bid in b.auction:
                if hasattr(bid, 'denom'):
                    bid_type.append('Contract')
                    denom.append(bid.denom.name)
                    penalty.append(None)
                    level.append(bid.level)
                    alertable.append(bid.alertable)
                    announcement.append(bid.announcement)
                else:
                    bid_type.append('Penalty')
                    denom.append(None)
                    penalty.append(bid.penalty.name)
                    level.append(None)
                    alertable.append(bid.alertable)
                    announcement.append(bid.announcement)
            board_d['bid_type'].append(bid_type)
            board_d['bid_denom'].append(denom)
            board_d['bid_penalty'].append(penalty)
            board_d['bid_level'].append(level)
            board_d['bid_alertable'].append(alertable)
            board_d['bid_announcement'].append(announcement)
            play_rank = []
            play_suit = []
            for play in b.play:
                play_rank.append(play.rank.name)
                play_suit.append(play.suit.name)
            board_d['play_rank'].append(play_rank)
            board_d['play_suit'].append(play_suit)

    # schema definitions for pl.DataFrame()
    schema = {
        'board_num':pl.UInt16,
        'dealer':pl.Utf8,
        'vulnerability':pl.Utf8,
        'passout':pl.Boolean,
        'contract':pl.Utf8, # drop_na later
        'level':pl.UInt8,
        'denom':pl.Utf8,
        'trump':pl.Utf8,
        'penalty':pl.Utf8,
        'declarer':pl.Utf8,
        'result':pl.Int8,
        'score':pl.Int16,
        'claimed':pl.Boolean,
        'PBN':pl.Utf8, # str later
        'Hand_N':pl.Utf8, # str later
        'Hand_E':pl.Utf8, # str later
        'Hand_S':pl.Utf8, # str later
        'Hand_W':pl.Utf8, # str later
        'info':pl.Struct, # unnest later
        'source_file':pl.Utf8,
        'bid_type':pl.List(pl.Utf8),
        'bid_denom':pl.List(pl.Utf8),
        'bid_penalty':pl.List(pl.Utf8),
        'bid_level':pl.List(pl.UInt8),
        'bid_alertable':pl.List(pl.Boolean),
        'bid_announcement':pl.List(pl.Utf8), # strip later
        'play_rank':pl.List(pl.Utf8),
        'play_suit':pl.List(pl.Utf8),
    }
    assert set(board_d.keys()) == set(schema.keys()), set(board_d.keys()).symmetric_difference(set(schema.keys()))

    # TypeError: unexpected value while building Series of type String; found value of type Struct([Field { name: "ordering", dtype: Null }, Field { name: "name", dtype: String }, Field { name: "minwidth", dtype: String }, Field { name: "alignment", dtype: String }]): {null,"Denomination","2","R"}
    df = pl.DataFrame(board_d,strict=False) # ,schema=schema # todo: complains about Player if strict=True. haven't been able to isolate why.

    # Struct columns need to be unnested.
    if 'info' in df.columns:
        df = df.unnest('info') # if derived from lin files: unnest 'info' (Struct(5)) into Player_[NESW] columns.
        if 'ScoreTable' in df.columns:
            st_dfs = []
            for i,b in enumerate(boards_in_lin_file):
                st_df = pl.DataFrame(b.info.ScoreTable['rows'],schema=[h['name'] for h in b.info.ScoreTable['headers']],orient="row")
                st_df = st_df.with_columns(pl.lit(b.board_num).alias('board_num'))
                st_df = st_df.select(['board_num'] + [col for col in st_df.columns if col != 'board_num'])
                st_dfs.append(st_df)
            st_df = pl.concat(st_dfs)
            # Explode the main DataFrame by joining with ScoreTable data
            df = df.join(st_df, on='board_num', how='inner')
            # todo: use exploded columns ['Contract', 'Declarer', 'Result'] to replace 'denom', 'penalty', 'trump', 'contract', 'level', 'bid_denom', 'bid_penalty', 'bid_trump', 'bid_contract', 'bid_level'

    # rename columns. some derive from lin->endplay, others from pbn->endplay
    # todo: after renaming, cast to preferred type.
    custom_naming_exceptions = {
        'BCFlags':'BCFlags',
        'Date':'Date',
        'Event':'Event',
        'Room':'Room',
        'Score':'Score',
        'Scoring':'Scoring',
        'Site':'Site',
        'North':'Player_N', # pl.String
        'East':'Player_E', # pl.String
        'South':'Player_S', # pl.String
        'West':'Player_W' # pl.String
    }
    df = df.rename(custom_naming_exceptions)

    # obsolete? isn't needed for lin but might be needed for pbn.
    # if i == 0: # first board
    #     custom_c_l = set(b.info.keys()).difference(boards_d.keys())
    #     #print(f'Creating columns for custom info keys: {custom_c_l}')
    # else: # subsequent boards must match first board's info keys
    #     custom_c_diffs = set(b.info.keys()).symmetric_difference(custom_c_l)
    #     assert custom_c_diffs == set(), custom_c_diffs
    # for c in custom_c_l:
    #     board_d[custom_naming_exceptions.get(c,'Custom_'+c)].append(b.info[c])

    # probably need to filter out rows. possibly bad practice to filter here because it changes row count?
    # filter out pbn of N:... ... ... ... and contract of null.
    # df = df.filter(pl.col('PBN').ne('N:... ... ... ...'))
    # df = df.filter(pl.col('contract').is_not_null())

    return df


# convert lin file columns to conform to bidding table columns.

# make sure the dicts have same dtypes for keys and values. It's required for some polars operations.

# all these dicts have been copied to mlBridgeLib.py. todo: remove these but requires using import mlBridgeLib.
Direction_to_NESW_d = {
    0:'N',
    1:'E',
    2:'S',
    3:'W',
    '0':'N',
    '1':'E',
    '2':'S',
    '3':'W',
    'north':'N',
    'east':'E',
    'south':'S',
    'west':'W',
    'North':'N',
    'East':'E',
    'South':'S',
    'West':'W',
    'N':'N',
    'E':'E',
    'S':'S',
    'W':'W',
    'n':'N',
    'e':'E',
    's':'S',
    'w':'W',
    None:None, # PASS
    '':'' # PASS
}

Strain_to_CDHSN_d = {
    'spades':'S',
    'hearts':'H',
    'diamonds':'D',
    'clubs':'C',
    'Spades':'S',
    'Hearts':'H',
    'Diamonds':'D',
    'Clubs':'C',
    'nt':'N',
    '♠':'S',
    '♥':'H',
    '♦':'D',
    '♣':'C',
    'NT':'N',
    'p':'PASS',
    'Pass':'PASS',
    'PASS':'PASS'
}

# todo: use mlBridgeLib.Vulnerability_to_Vul_d instead?
Vulnerability_to_Vul_d = {
    0: 'None',
    1: 'N_S',
    2: 'E_W',
    3: 'Both',
    '0': 'None',
    '1': 'N_S',
    '2': 'E_W',
    '3': 'Both',
    'None': 'None',
    'N_S': 'N_S',
    'E_W': 'E_W',
    'N-S': 'N_S',
    'E-W': 'E_W',
    'Both': 'Both',
    'NS': 'N_S',
    'EW': 'E_W',
    'All': 'Both',
    'none': 'None',
    'ns': 'N_S',
    'ew': 'E_W',
    'both': 'Both',
}

EpiVul_to_Vul_NS_Bool_d = {
    0: False,
    1: True,
    2: False,
    3: True,
}

EpiVul_to_Vul_EW_Bool_d = {
    0: False,
    1: False,
    2: True,
    3: True,
}

Dbl_to_X_d = {
    'passed':'',
    'doubled':'X',
    'redoubled':'XX',
    'p':'',
    'd':'X',
    'r':'XX',
    'p':'',
    'x':'X',
    'xx':'XX'
}


def convert_endplay_df_to_mlBridge_df(df):

    # with_columns() is separated for easier debugging. iVul has to be separated anyways.
    # todo: create Date column using date embedded in source file name.
    # todo: is pair number ns, pair number ew needed?
    # todo: make relevant columns categorical.

    if 'index' not in df.columns:
        df = df.with_row_index() # useful for keeping track of rows.

    df = df.with_columns(
        pl.Series('Board',df['board_num'],pl.UInt8),
    )
    df = df.with_columns(
        pl.Series('Dealer', [Direction_to_NESW_d[d] for d in df['dealer']], pl.String, strict=False), # todo: using list comprehension instead of type error when using replace_strict().
    )
    df = df.with_columns(
        pl.Series('Vul', [Vulnerability_to_Vul_d[v] for v in df['vulnerability']], pl.String, strict=False), # todo: using list comprehension instead of type error when using replace_strict().
    )
    df = df.with_columns(
        pl.Series('iVul',df['vulnerability'].cast(pl.UInt8),pl.UInt8), # assumes input is 0 or '0', etc.
    )
    df = df.with_columns(
        pl.col('iVul').replace_strict(EpiVul_to_Vul_NS_Bool_d,return_dtype=pl.Boolean).alias('Vul_NS'),
    )
    df = df.with_columns(
        pl.col('iVul').replace_strict(EpiVul_to_Vul_EW_Bool_d,return_dtype=pl.Boolean).alias('Vul_EW'),
    )
    # BridgeWeb uses ScoreTable.
    if 'ScoreTable' in df.columns:
        df = df.with_columns(
           pl.col('Contract').str.replace('NT','N').str.extract(r"^([^-+]*)", 1),
        )
        df = df.with_columns(
            pl.Series('Dbl',df['Contract'].str.slice(2),pl.String), # categorical, yes
        )
        df = df.with_columns(
            pl.col('Contract').str.slice(0,1).cast(pl.UInt8, strict=False).alias('BidLvl'), # Extract first character (bid level)
        )
        df = df.with_columns(
            pl.col('Contract').str.slice(1,1).alias('BidSuit'), # Extract second character (suit) from contract (categorical, yes)
        )
        df = df.with_columns(
            # easier to use discrete replaces instead of having to slice contract (nt, pass would be a complication)
            # first NT->N and suit symbols to SHDCN
            # If BidLvl is None, make Contract None
            pl.when(pl.col('BidLvl').is_null())
            .then(None)
            .otherwise(pl.col('BidLvl').cast(pl.String)+pl.col('BidSuit')+pl.col('Dbl')+pl.col('Declarer'))
            .alias('Contract'),
        )
        df = df.with_columns(
            pl.Series('trump',df['trump'].replace_strict(Strain_to_CDHSN_d,return_dtype=pl.String),pl.String),# categorical?
        )
        df = df.with_columns(
            pl.Series('Declarer_Direction', [Direction_to_NESW_d[d] for d in df['Declarer']], pl.String, strict=False), # todo: using list comprehension instead of type error when using replace_strict().
        )
        df = df.with_columns(
            pl.Series('Tricks',df['Result'].cast(pl.UInt8, strict=False).fill_nan(0),pl.UInt8),
        )
        # example of creating unneeded column. could drop here and leave it to mlBridgeAugmentLib to recreate with proper values.
        df = df.with_columns(
            pl.Series('Result',df['Tricks'].cast(pl.Int8, strict=False)-6-df['BidLvl']), # overwrites Result column.
        )
        # leave it to mlBridgeAugmentLib to convert contract parts to Score and Score_NS.
        # df = df.with_columns(
        #     pl.Series('Score',df['score'].cast(pl.Int16, strict=False).fill_nan(0),pl.Int16),
        # )
        df = df.with_columns(
            pl.col('PairId_NS').cast(pl.Utf8).alias('Pair_Number_NS'), # todo: fake Pair_Number_NS
        )
        df = df.with_columns(
            pl.col('PairId_EW').cast(pl.Utf8).alias('Pair_Number_EW'), # todo: fake Pair_Number_EW
        )
        df = df.with_columns(
            pl.Series('Player_ID_N',df['Pair_Number_NS']+'_N',pl.String), # todo: fake player id
        )
        df = df.with_columns(
            pl.Series('Player_ID_E',df['Pair_Number_EW']+'_E',pl.String), # todo: fake player id
        )
        df = df.with_columns(
            pl.Series('Player_ID_S',df['Pair_Number_NS']+'_S',pl.String), # todo: fake player id
        )
        df = df.with_columns(
            pl.Series('Player_ID_W',df['Pair_Number_EW']+'_W',pl.String), # todo: fake player id
        )
        # todo: rename ScorePercent_NS to MP_Pct_NS?
        # df = df.with_columns(
        #     pl.Series('?',df['ScorePercent_NS'],pl.Float64),
        # )
        # df = df.with_columns(
        #     pl.Series('?',df['ScorePercent_EW'],pl.Float64),
        # )
        # todo: rename MatchPoints_NS to ?
        # df = df.with_columns(
        #     pl.Series('?',df['MatchPoints_NS'],pl.Float64),
        # )
        # df = df.with_columns(
        #     pl.Series('?',df['MatchPoints_EW'],pl.Float64),
        # )
    else:
        #pl.Series('passout',df['passout'],pl.Boolean), # todo: make passout a boolean in previous step.
        df = df.with_columns(
            # easier to use discrete replaces instead of having to slice contract (nt, pass would be a complication)
            # first NT->N and suit symbols to SHDCN
            pl.Series('Contract',df['contract'],pl.String).str.replace('NT','N').str.replace('♠','S').str.replace('♥','H').str.replace('♦','D').str.replace('♣','C').str.extract(r"^([^-+]*)", 1),
        )
        df = df.with_columns(
            pl.Series('BidLvl',df['level'].cast(pl.UInt8, strict=False),pl.UInt8), # todo: make level a uint8 in previous step.
        )
        df = df.with_columns(
            pl.Series('BidSuit',df['denom'].replace_strict(Strain_to_CDHSN_d,return_dtype=pl.String),pl.String),# categorical, yes
        )
        df = df.with_columns(
            pl.Series('trump',df['trump'].replace_strict(Strain_to_CDHSN_d,return_dtype=pl.String),pl.String),# categorical?
        )
        df = df.with_columns(
            pl.Series('Dbl',df['penalty'].replace_strict(Dbl_to_X_d,return_dtype=pl.String),pl.String),# categorical, yes
        )
        df = df.with_columns(
            pl.Series('Declarer_Direction', [Direction_to_NESW_d[d] for d in df['declarer']], pl.String, strict=False), # todo: using list comprehension instead of type error when using replace_strict().
        )
        df = df.with_columns(
            pl.Series('Result',df['result'].cast(pl.Int8, strict=False).fill_nan(0),pl.Int8),
        )
        df = df.with_columns(
            pl.Series('Tricks',df['level'].cast(pl.Int8, strict=False).fill_nan(0)+df['result'].cast(pl.Int8, strict=False).fill_nan(0)+6,pl.UInt8),
        )
        df = df.with_columns(
            pl.Series('Score',df['score'].cast(pl.Int16, strict=False).fill_nan(0),pl.Int16),
        )
        df = df.with_columns(
            pl.lit(0).cast(pl.UInt32).alias('Pair_Number_NS'), # todo: fake Pair_Number_NS
        )
        df = df.with_columns(
            pl.lit(0).cast(pl.UInt32).alias('Pair_Number_EW'), # todo: fake Pair_Number_EW
        )
        df = df.with_columns(
            pl.Series('Player_Name_N',df['Player_N'],pl.String),
        )
        df = df.with_columns(
            pl.Series('Player_Name_E',df['Player_E'],pl.String),
        )
        df = df.with_columns(
            pl.Series('Player_Name_S',df['Player_S'],pl.String),
        )
        df = df.with_columns(
            pl.Series('Player_Name_W',df['Player_W'],pl.String),
        )
        df = df.with_columns(
            pl.Series('Player_ID_N',df['Pair_Number_NS']+'_N',pl.String), # todo: fake player id
        )
        df = df.with_columns(
            pl.Series('Player_ID_E',df['Pair_Number_EW']+'_E',pl.String), # todo: fake player id
        )
        df = df.with_columns(
            pl.Series('Player_ID_S',df['Pair_Number_NS']+'_S',pl.String), # todo: fake player id
        )
        df = df.with_columns(
            pl.Series('Player_ID_W',df['Pair_Number_EW']+'_W',pl.String), # todo: fake player id
        )
    df = df.with_columns(
        df['claimed'].cast(pl.Boolean, strict=False),
    )
    df = df.with_columns(
        pl.Series('source_file',df['source_file'],pl.String),
    )
    df = df.with_columns(
        pl.Series('bid_type',df['bid_type'],pl.List(pl.String)), # categorical?
    )
    df = df.with_columns(
        pl.Series('bid_denom',df['bid_denom'],pl.List(pl.String)), # categorical?
    )
    df = df.with_columns(
        pl.Series('bid_penalty',df['bid_penalty'],pl.List(pl.String)), # categorical?
    )
    df = df.with_columns(
        #pl.Series('bid_level',df['bid_level'].cast(pl.List(pl.Int64), strict=False)+1,pl.List(pl.Int64)), # todo: make bid_level a uint8 in previous step.
    )
    df = df.with_columns(
        pl.Series('bid_alertable',df['bid_alertable'],pl.List(pl.Boolean)),
    )
    df = df.with_columns(
        pl.Series('bid_announcement',df['bid_announcement'],pl.List(pl.String)),
    )
    df = df.with_columns(
        pl.Series('play_rank',df['play_rank'],pl.List(pl.String)),# categorical?
    )
    df = df.with_columns(
        pl.Series('play_suit',df['play_suit'],pl.List(pl.String)),# categorical?
    )
    # drop unused or obsolete columns
    df = df.drop(
        {
            'board_num',
            'dealer',
            'vulnerability',
            'contract',
            'level',
            'denom',
            'penalty',
            'declarer',
            'result',
            'score',
            'Player_N',
            'Player_E',
            'Player_S',
            'Player_W',
            #'ScorePercent_NS',
            #'MatchPoints_NS',
            #'ScorePercent_EW',
            #'MatchPoints_EW',
        }
    )
    return df
