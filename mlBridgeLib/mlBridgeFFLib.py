import polars as pl
from endplay.types import Deal # only used to correct and validate pbns. pbn == Deal(pbn).to_pbn()

iVulToVul_d = { # todo: make enum?
    0:'None',
    1:'N_S',
    2:'E_W',
    3:'Both',
}
FrenchPairDirectionToPairDirection_d = { # convert mlBridgeLib dealer to endplay dealer
    'NS':'NS',
    'EW':'EW',
    'EO':'EW',
}
FrenchDirectionToDirection_d = { # convert mlBridgeLib dealer to endplay dealer
    'N':'N',
    'E':'E',
    'S':'S',
    'W':'W',
    'O':'W',
    '-':None, # happened when computer failed to communicate with bridgemates.
    None:None,  # Map 'Unknown' to None/NULL
}
FrenchStrainToStrain_d = { # convert mlBridgeLib dealer to endplay dealer
    'P':'S',
    'C':'H', # oh dear, ambiguous in English (cour vs club). Can only be used if totally French input.
    'K':'D',
    'T':'C',
    'SA':'N', # needed to make single character strain symbol
    'N':'N', # needed to make single character strain symbol
}
FrenchCardsToCards_d = {
    'A':'A',
    'R':'K',
    'D':'Q',
    'V':'J',
    '0':'T',
    '9':'9',
    '8':'8',
    '7':'7',
    '6':'6',
    '5':'5',
    '4':'4',
    '3':'3',
    '2':'2',
}
FrenchVulToVul_d = {
    'Personne':'None',
    'NS':'N_S',
    'EO':'E_W',
    'Tous':'Both',
    'None':'None',
    'N_S':'N_S',
    'E_W':'E_W',
    'Both':'Both',
    'Unknown':None,  # Map 'Unknown' to None/NULL
}

def BoardNumberToDealer(bn):
    return 'NESW'[(bn-1) & 3]

def BoardNumberToiVul(bn):
    bn -= 1
    return range(bn//4, bn//4+4)[bn & 3] & 3

def BoardNumberToVul(bn):
    return iVulToVul_d[BoardNumberToiVul(bn)]

def PbnToN(bd):
    hands = bd[2:].split(' ')
    d = bd[0]
    match d:
        case 'N':
            pbn = bd
        case 'E':
            pbn = 'N:'+' '.join([hands[1],hands[2],hands[3],hands[0]])
        case 'S':
            pbn = 'N:'+' '.join([hands[2],hands[3],hands[0],hands[1]])
        case 'W':
            pbn = 'N:'+' '.join([hands[3],hands[0],hands[1],hands[2]])
        case _:
            raise ValueError(f"Invalid dealer: {d}")
    dpbn = Deal(pbn).to_pbn()
    if pbn != dpbn:
        print(f"Invalid PBN: {pbn} != {dpbn}") # often a sort order issue.
    return dpbn

def FrenchCardsToPBN(df):
    directions = ['north', 'east', 'south', 'west']
    suits = ['Spade', 'Heart', 'Diamond', 'Club']

    # Create PBN string with suit separators and French->English card conversion
    direction_parts = []
    for direction in directions:
        direction_part = pl.concat_str([
            pl.col(f'deal_{direction}Card{suit}').map_elements(
                lambda s: ''.join([FrenchCardsToCards_d.get(c, c) for c in s]), 
                return_dtype=pl.Utf8
            ) for suit in suits
        ], separator='.')
        direction_parts.append(direction_part)
    
    return df.with_columns(pl.concat_str([pl.lit('N:'), pl.concat_str(direction_parts, separator=' ')]).alias('PBN'))

def convert_ffdf_api_to_mldf(ffldfs):

    # simultaneous_tournaments columns:
    # ['simultane_id', 'nb_days_blocked_results', 'date', 'type', 'type_code', 'name', 'moment', 'moment_code', 'code', 'co_organizer_name',
    # 'co_organizer_type', 'nb_total_pairs', 'is_homo', 'simultaneeCode', 'simultaneeId', 'season_id', 'has_general_ranking', 'deal_count',
    # 'is_imp', 'calcul_mode', 'website', 'website_orga', 'team_PE', 'team_PE_bonus', 'team_id', 'team_organization_code', 'team_organization_id',
    # 'team_organization_name', 'team_orientation', 'team_percent', 'team_players_firstname', 'team_players_gender', 'team_players_id',
    # 'team_players_is_licensee', 'team_players_is_suspended', 'team_players_lastname', 'team_players_license_number', 'team_players_position',
    # 'team_players_team_id', 'team_position', 'team_ranking', 'team_section_name', 'team_table_number', 'team_theoretical_ranking']

    # simultaneous_deals columns:
    # ['team_ns_id', 'team_eo_id', 'section', 'contract', 'declarant','first_card', 'result', 'score_ns', 'score_eo', 'note_ns', 'note_eo', 'Board']

    # simultaneous_dealsNumber columns:
    # ['nb_deals']

    # simultaneous_roadsheets columns:
    # ['roadsheets_deals_contract', 'roadsheets_deals_dealNumber', 'roadsheets_deals_declarant', 'roadsheets_deals_first_card', 'roadsheets_deals_opponentsAvgNote', 'roadsheets_deals_opponentsNote', 'roadsheets_deals_opponentsOrientation',
    # 'roadsheets_deals_opponentsScore', 'roadsheets_deals_result', 'roadsheets_deals_teamAvgNote', 'roadsheets_deals_teamNote',
    # 'roadsheets_deals_teamOrientation','roadsheets_deals_teamScore', 'roadsheets_teams_cpt',
    # 'roadsheets_player_n', 'roadsheets_player_s', 'roadsheets_player_e', 'roadsheets_player_w']

    # simultaneous_deals columns:
    # ['frequencies_dealNumber', 'frequencies_doubleTopageNumber', 'frequencies_noteEO', 'frequencies_noteNS', 'frequencies_organizations_code',
    # 'frequencies_organizations_name', 'frequencies_organizations_organization_id', 'frequencies_scoreEO', 'frequencies_scoreFrequency',
    # 'frequencies_scoreNS', 'frequencies_sectionNumber', 'frequencies_selected', 'frequencies_topValue', 'nb_days_blocked_results',
    # 'deal_dealNumber', 'deal_northCardSpade', 'deal_northCardHeart', 'deal_northCardDiamond', 'deal_northCardClub',
    # 'deal_eastCardSpade', 'deal_eastCardHeart', 'deal_eastCardDiamond', 'deal_eastCardClub',
    # 'deal_southCardSpade', 'deal_southCardHeart', 'deal_southCardDiamond', 'deal_southCardClub',
    # 'deal_westCardSpade', 'deal_westCardHeart', 'deal_westCardDiamond', 'deal_westCardClub',
    # 'deal_vulnerability', 'deal_dealer', 'tournament_type', 'tournament_code', 'tournament_moment', 'tournament_date',
    # 'tournament_name', 'tournament_top', 'tournament_rank_type', 'tournament_organization_name', 'tournament_team_section_position',
    # 'tournament_team_section_table_number', 'tournament_team_section_name', 'tournament_is_imp', 'tournament_nb_pairs',
    # 'teams_players_name_firstname', 'teams_players_name_gender', 'teams_players_name_id', 'teams_players_name_lastname',
    # 'teams_players_position', 'teams_players_rank', 'teams_players_percent', 'teams_opponents_name_firstname',
    # 'teams_opponents_name_gender', 'teams_opponents_name_lastname', 'teams_opponents_name_oppo_id', 'teams_opponents_position']

    st_df = ffldfs['simultaneous_tournaments_by_organization_id']
    cols = ['team_organization_code', 'team_organization_id', 'team_section_name',
            'team_orientation', 'team_table_number', 'team_players_position',
            'team_players_id', 'team_players_firstname', 'team_players_lastname']
    st_df = st_df[cols].unique() # st_df was exploded so must now deduplicate. height should be 4 x number of simultaneous pairs.
    #player_to_pair_d = dict(zip(st_df['team_players_id'],st_df[('team_section_name','team_orientation','team_table_number')].rows()))
    #pair_number_to_player_d = dict(zip(st_df[('team_section_name','team_orientation','team_table_number')].rows(),st_df['team_players_id']))

    sd_df = ffldfs['simultaneous_deals'] # todo: use hand record info e.g. deal_vulnerability, deal_dealer, deal_dealNumber, tournament_team_section_position
    sd_df = sd_df.filter(pl.col('teams_players_position').ne(pl.col('teams_opponents_position'))) # hack to drop unplayed boards. Otherwise boards become Boards_I_Played.
    assert sd_df.height%4 == 0, 'simultaneous_deals must have 4 rows per board. One for each direction.'
    sd_df = FrenchCardsToPBN(sd_df) # convert hand columns to PBN here before the columns are removed.
    sd_df = sd_df.with_columns([
        pl.Series([d for i in range(sd_df.height//4) for d in '1234']).alias('Player_Direction'), # will map indexes to directions later.
    ])
    sd_df = sd_df.with_columns([
        pl.col('frequencies').list.get(0).struct.field('topValue').cast(pl.UInt32).alias('MP_Top'), # todo: problem. only works for common game. might fail otherwise?
    ])
    cols = [
        'PBN', 'Player_Direction', 'MP_Top',
        'tournament_date', 'tournament_team_section_name', 'tournament_team_section_table_number',
        'deal_dealNumber', 'deal_dealer', 'deal_vulnerability',
        'teams_players_position', 'teams_opponents_position',
        'teams_players_name_id', 'teams_opponents_name_oppo_id'
    ]
    sd_df = sd_df[cols].unique() # sd_df was exploded so must now deduplicate

    sd_df = sd_df.with_columns([
        pl.col('tournament_date').alias('Date'),
    ])
 
    sd_df = sd_df.with_columns([
        pl.col('tournament_team_section_name').alias('Section_Name'),
    ])

    sd_df = sd_df.with_columns([
        pl.col('teams_players_position').replace_strict(FrenchPairDirectionToPairDirection_d, return_dtype=pl.Utf8).alias('Pair_Direction'),
    ])

    sd_df = sd_df.with_columns([
        pl.col('teams_opponents_position').replace_strict(FrenchPairDirectionToPairDirection_d, return_dtype=pl.Utf8).alias('Opponent_Pair_Direction'),
    ])
 
    sd_df = sd_df.with_columns([
        pl.col('tournament_team_section_table_number').cast(pl.UInt32).alias('Pair_Number'),
    ])

    sd_df = sd_df.with_columns([
        pl.col('deal_dealNumber').cast(pl.UInt32).alias('Board'),
    ])

    sd_df = sd_df.with_columns([
        pl.col('deal_dealer').replace_strict(FrenchDirectionToDirection_d, return_dtype=pl.Utf8).alias('Dealer'),
    ])

    sd_df = sd_df.with_columns([
        pl.col('deal_vulnerability').replace_strict(FrenchVulToVul_d, return_dtype=pl.Utf8).alias('Vul'),
    ])

    sd_df = sd_df.with_columns([
        pl.col('deal_vulnerability').replace_strict(FrenchVulToVul_d, return_dtype=pl.Utf8).alias('Vul'),
    ])

   # reduce height to 1/4 of original (only the boards played by home pair) by removing non-unique columns.
    sd_df = sd_df['Date','Section_Name','Board','PBN','Dealer','Vul','Pair_Direction','Pair_Number','MP_Top'].unique().sort('Board')

    if sd_df['Pair_Direction'].eq('NS').all():
        pair_direction = 'NS'
    elif sd_df['Pair_Direction'].eq('EW').all():
        pair_direction = 'EW'
    else:
        raise ValueError(f"Invalid Pair_Direction: {sd_df['Pair_Direction'].unique()}. Howell movements are not implemented. Try a different game.")

    # df = ffldfs['simultaneous_roadsheets']

    # df = df.with_columns([
    #     pl.col('roadsheets_deals_dealNumber').cast(pl.UInt32).alias('Board'), # cast and rename to Board
    # ])

    # # result height should be equal to the number of boards played by home pair.
    # df = sd_df.join(df,on='Board',how='inner').unique().sort('Board')

    # result height should be equal to the number of boards played at all tables.
    # todo: renaming is only needed when dealing with roadsheet naming conventions. But we've moved away from roadsheets so maybe should use simultaneous_description_by_organization_id names.
    simultaneous_description_by_organization_id_df = ffldfs['simultaneous_description_by_organization_id']
    simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
        pl.col('section').alias('Section_Name'),
    ]).drop('section')
    simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
        pl.col('contract').alias('roadsheets_deals_contract'),
    ]).drop('contract')
    simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
        pl.col('declarant').alias('roadsheets_deals_declarant'),
    ]).drop('declarant')
    simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
        pl.col('first_card').alias('roadsheets_deals_first_card'),
    ]).drop('first_card')
    simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
        pl.col('result').alias('roadsheets_deals_result'),
    ]).drop('result')
    if pair_direction == 'NS':
        simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
            pl.col('score_ns').alias('roadsheets_deals_teamScore'),
        ]).drop('score_ns')
        simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
            pl.col('score_eo').alias('roadsheets_deals_opponentsScore'),
        ]).drop('score_eo')
        simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
            pl.col('note_ns').alias('roadsheets_deals_teamAvgNote'),
        ]).drop('note_ns')
        simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
            pl.col('note_eo').alias('roadsheets_deals_opponentsAvgNote'),
        ]).drop('note_eo')
    elif pair_direction == 'EW':
        simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
            pl.col('score_ns').alias('roadsheets_deals_opponentsScore'),
        ]).drop('score_ns')
        simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
            pl.col('score_eo').alias('roadsheets_deals_teamScore'),
        ]).drop('score_eo')
        simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
            pl.col('note_ns').alias('roadsheets_deals_opponentsAvgNote'),
        ]).drop('note_ns')
        simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
            pl.col('note_eo').alias('roadsheets_deals_teamAvgNote'),
        ]).drop('note_eo')
    else:
        raise ValueError(f"Invalid Pair_Direction: {sd_df['Pair_Direction'].unique()}")
    simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
        pl.col('team_ns_id').alias('Pair_Number_NS'),
    ]).drop('team_ns_id')
    simultaneous_description_by_organization_id_df = simultaneous_description_by_organization_id_df.with_columns([
        pl.col('team_eo_id').alias('Pair_Number_EW'),
    ]).drop('team_eo_id')

    df = simultaneous_description_by_organization_id_df.join(
        sd_df,
        on=['Section_Name','Board'], how='inner'
    ).sort('Board')
    
    # Create lookup dictionaries from st_df using one-liners
    # First create the full name column
    full_names = st_df.with_columns([
        (pl.col('team_players_firstname') + pl.lit(' ') + pl.col('team_players_lastname')).alias('full_name')
    ])['full_name']
    
    player_id_lookup = dict(zip(
        st_df[('team_section_name','team_orientation','team_players_position','team_table_number')].rows(),
        st_df['team_players_id'].cast(pl.Utf8)
    ))

    player_name_lookup = dict(zip(
        st_df['team_section_name','team_orientation','team_players_position','team_table_number'].rows(),
        full_names # cannot be done directly. must convert to series or list.
    ))
    
    print(f"Created player lookups with {len(player_id_lookup)} entries")
    
    # Define player mappings
    player_mappings = [
        ('NS', 1, 'Pair_Number_NS', 'N'),  # North: NS orientation, position 1
        ('EW', 1, 'Pair_Number_EW', 'E'),  # East: EW orientation, position 1
        ('NS', 2, 'Pair_Number_NS', 'S'),  # South: NS orientation, position 2
        ('EW', 2, 'Pair_Number_EW', 'W')   # West: EW orientation, position 2
    ]
    
    # Add player ID and name columns using dictionary lookup
    for orientation, position, pair_col, direction in player_mappings:
        # Add columns using the lookup dictionaries
        df = df.with_columns([
            pl.struct(['Section_Name', pair_col]).map_elements(
                lambda x: player_id_lookup.get((x['Section_Name'], orientation, position, x[pair_col]), None), 
                return_dtype=pl.Utf8
            ).alias(f'Player_ID_{direction}'),
            pl.struct(['Section_Name', pair_col]).map_elements(
                lambda x: player_name_lookup.get((x['Section_Name'], orientation, position, x[pair_col]), None), 
                return_dtype=pl.Utf8
            ).alias(f'Player_Name_{direction}')
        ])

    df = df.with_columns([
        pl.col('roadsheets_deals_declarant').replace_strict(FrenchDirectionToDirection_d, return_dtype=pl.Utf8).alias('Declarer'),
    ])

    df = df.with_columns([
        pl.when((pl.col('roadsheets_deals_contract').str.contains(r'^[1-7]'))) # begins with 1-7 (level)
            .then(
                pl.concat_str([
                    pl.col('roadsheets_deals_contract').str.slice(0,1), # level
                    # default='?' is required to handle PASS even though it's only used by .then(pl.lit('PASS'))
                    pl.col('roadsheets_deals_contract').str.replace('SA', 'N').str.slice(1,1).replace_strict(FrenchStrainToStrain_d, default='?', return_dtype=pl.Utf8), # strain
                    pl.col('roadsheets_deals_contract').str.replace('SA', 'N').str.replace('x', 'X').str.slice(2), # double
                    pl.col('Declarer'), # declarer
                ]))
            .when(pl.col('roadsheets_deals_contract').str.to_uppercase().str.starts_with('PASS')) # e.g. 'PASS' or 'passe' or 'PASSE'
            .then(pl.lit('PASS'))
            .otherwise(None) # catch all for invalid contracts.
            .alias('Contract'),
    ])
    
    df = df.with_columns([
        pl.when(pl.col('roadsheets_deals_result').str.starts_with('+'))
            .then(pl.col('roadsheets_deals_result').str.slice(1))  # Remove '+'
            .when(pl.col('roadsheets_deals_result').eq('-')) # happened when computer failed to communicate with bridgemates.
            .then(None)
            .when(pl.col('roadsheets_deals_result').str.starts_with('-'))
            .then(pl.col('roadsheets_deals_result'))
            .otherwise(pl.lit('0'))  # Replace '=' with '0'
            .cast(pl.Int8)
            .alias('Result'),
    ])

    # todo: need 'if' test for different values (team home/opponents) to create NS/EW/EO aliases. otherwise the same code.
    # todo: debug not all pairs are NS or EW.
    if pair_direction == 'NS':
        df = df.with_columns([
            pl.when(pl.col('roadsheets_deals_teamScore').str.contains(r'^\d+$'))
                .then(pl.col('roadsheets_deals_teamScore'))
                .when(pl.col('roadsheets_deals_teamScore').str.to_uppercase().str.starts_with('PASS')) # e.g. 'PASS' or 'passe' or 'PASSE'
                .then(pl.lit('0'))
                .when(pl.col('roadsheets_deals_opponentsScore').str.contains(r'^\d+$'))
                .then('-'+pl.col('roadsheets_deals_opponentsScore'))
                .otherwise(pl.lit(None))
                .cast(pl.Int16)
                .alias('Score_NS'),
        ])
        df = df.with_columns([
            pl.when(pl.col('roadsheets_deals_opponentsScore').str.contains(r'^\d+$'))
                .then(pl.col('roadsheets_deals_opponentsScore'))
                .when(pl.col('roadsheets_deals_opponentsScore').str.to_uppercase().str.starts_with('PASS')) # e.g. 'PASS' or 'passe' or 'PASSE'
                .then(pl.lit('0'))
                .when(pl.col('roadsheets_deals_teamScore').str.contains(r'^\d+$'))
                .then('-'+pl.col('roadsheets_deals_teamScore'))
                .otherwise(pl.lit(None))
                .cast(pl.Int16)
                .alias('Score_EW'),
        ])
    elif pair_direction == 'EW':
        df = df.with_columns([
            pl.when(pl.col('roadsheets_deals_teamScore').str.contains(r'^\d+$'))
                .then(pl.col('roadsheets_deals_teamScore'))
                .when(pl.col('roadsheets_deals_teamScore').str.to_uppercase().str.starts_with('PASS')) # e.g. 'PASS' or 'passe' or 'PASSE'
                .then(pl.lit('0'))
                .when(pl.col('roadsheets_deals_opponentsScore').str.contains(r'^\d+$'))
                .then('-'+pl.col('roadsheets_deals_opponentsScore'))
                .otherwise(pl.lit(None))
                .cast(pl.Int16)
                .alias('Score_EW'),
        ])
        df = df.with_columns([
            pl.when(pl.col('roadsheets_deals_opponentsScore').str.contains(r'^\d+$'))
                .then(pl.col('roadsheets_deals_opponentsScore'))
                .when(pl.col('roadsheets_deals_opponentsScore').str.to_uppercase().str.starts_with('PASS')) # e.g. 'PASS' or 'passe' or 'PASSE'
                .then(pl.lit('0'))
                .when(pl.col('roadsheets_deals_teamScore').str.contains(r'^\d+$'))
                .then('-'+pl.col('roadsheets_deals_teamScore'))
                .otherwise(pl.lit(None))
                .cast(pl.Int16)
                .alias('Score_NS'),
        ])
    else:
        raise ValueError(f"Invalid Pair_Direction: {sd_df['Pair_Direction'].unique()}")
    
    # Ignoring existing simultaneous MP_Top column. Use window function over Board count to calculate MP_Top.
    # Could have issues with director adjustments.
    df = df.with_columns([
            (pl.col('Board').count().over('Board') - 1).alias('MP_Top')
        ])

    # Ignoring existing simultaneous MP_(NS|EW) columns. Use window function over Board count to calculate MP_NS. MP_EW is derived from MP_Top - MP_NS.
    # Could have issues with director adjustments.
    df = df.with_columns([
        (pl.col('Score_NS').rank().over('Board') - 1).alias('MP_NS'),
    ])
    df = df.with_columns([
        (pl.col('MP_Top') - pl.col('MP_NS')).alias('MP_EW'),
    ])

    # Ignoring existing simultaneous Pct_(NS|EW) columns. Calculate by using MP_Top.
    # Could have issues with director adjustments.
    df = df.with_columns([
        (pl.col('MP_NS')/pl.col('MP_Top')).alias('Pct_NS'),
    ])
    df = df.with_columns([
        (1 - pl.col('Pct_NS')).alias('Pct_EW'),
    ])

    return df


# this function uses lancelot api to create mldf.
# todo: update with newer algorithms from convert_ffdf_to_mldf().
def convert_ffdf_lancelot_to_mldf(ffdf):

    # assignments are broken into parts for polars compatibility (could be parallelized).
    #for col in ffdf.columns:
    #    if ((ffdf[col].dtype == pl.String) and ffdf[col].is_in(['PASS']).any()):
    #        print(col)

    df = ffdf.select([
        pl.col('group_id'),
        pl.col('board_id'),
        #pl.col('team_session_id'),
        #pl.col('team_id'),
        #pl.col('session_id'),
        pl.col('boardNumber').alias('Board'),
        #pl.col('board_frequencies'),
        # flatten the board_frequencies column into multiple columns
        # todo: need to generalize this to work for any json column with a list of structs. either do it here or in previous step.
        pl.col('board_frequencies').list.eval(pl.element().struct.field('nsScore')).alias('Scores_List_NS'),
        pl.col('board_frequencies').list.eval(pl.element().struct.field('nsNote')).alias('Pcts_List_NS'),
        pl.col('board_frequencies').list.eval(pl.element().struct.field('ewScore')).alias('Scores_List_EW'),
        pl.col('board_frequencies').list.eval(pl.element().struct.field('ewNote')).alias('Pcts_List_EW'),
        pl.col('board_frequencies').list.eval(pl.element().struct.field('count')).alias('Score_Freq_List'),
        # todo: have not solved the issue of which direction is dealer using ffbridge. I'm missing something but don't know what.
        # derive dealer from boardNumber which works for standard bridge boards but isn't guaranteed to work for all boards and events.
        pl.col('boardNumber')
            .map_elements(BoardNumberToDealer,return_dtype=pl.String)
            #.alias('DealerFromBoardNumber'),
            .alias('Dealer'),
        #pl.col('board_deal').str.slice(0, 1).alias('Dealer'),  # Use first character of 'board_deal' to create Dealer column.
        pl.col('board_deal')
            .map_elements(PbnToN,return_dtype=pl.String)
            .alias('PBN'),
        pl.col('boardNumber') # todo: check that there's no Vul already in the data.
            .map_elements(BoardNumberToiVul,return_dtype=pl.UInt8)
            .replace_strict(iVulToVul_d)
        .alias('Vul'),
        pl.when((pl.col('contract').str.contains(r'^[1-7]'))) # begins with 1-7 (level)
            .then(
                pl.concat_str([
                    pl.col('contract').str.slice(0,1), # level
                    pl.col('contract').str.replace('NT', 'N').str.slice(1,1), # strain
                    pl.col('contract').str.replace('NT', 'N').str.slice(2), # double
                    pl.col('declarer'),
                ]))
            .when(pl.col('contract').eq('PASS'))
            .then(pl.lit('PASS'))
            .otherwise(None) # catch all for invalid contracts.
            .alias('Contract'),
        pl.when(pl.col('result').str.starts_with('+'))
            .then(pl.col('result').str.slice(1))  # Remove '+'
            .when(pl.col('result').str.starts_with('-'))
            .then(pl.col('result'))
            .otherwise(pl.lit('0'))  # Replace '=' with '0'
            .cast(pl.Int16)
            .alias('Result'),
        # not liking that only one of the two columns (nsScore or ewScore) has a value. I prefer to have both with opposite signs.
        # although this may be an issue for director adjustments. Creating new columns (Score_NS and Score_EW) with opposite signs.
        pl.when(pl.col('nsScore').str.contains(r'^\d+$'))
            .then(pl.col('nsScore'))
            .when(pl.col('ewScore').str.contains(r'^\d+$'))
            .then('-'+pl.col('ewScore'))
            .otherwise(pl.lit(None))
            .cast(pl.Int16)
            .alias('Score_NS'),
        pl.when(pl.col('ewScore').str.contains(r'^\d+$'))
            .then(pl.col('ewScore'))
            .when(pl.col('nsScore').str.contains(r'^\d+$'))
            .then('-'+pl.col('nsScore'))
            .otherwise(pl.lit(None))
            .cast(pl.Int16)
            .alias('Score_EW'),
        (pl.col('nsNote')/100.0).alias('Pct_NS'),
        (pl.col('ewNote')/100.0).alias('Pct_EW'),
        # is this player1_id for every row table or just the requested team? remove until understood.
        # pl.col('team_player1_ffbId').alias('player1_id'),
        # pl.col('team_player1_firstName').alias('player1_firstName'),
        # pl.col('team_player1_lastName').alias('player1_lastName'),
        # pl.col('team_player2_ffbId').alias('player2_id'),
        # pl.col('team_player2_firstName').alias('player2_firstName'),
        # pl.col('team_player2_lastName').alias('player2_lastName'),
        (pl.col('lineup_northPlayer_firstName')+pl.lit(' ')+pl.col('lineup_northPlayer_lastName')).alias('Player_Name_N'),
        (pl.col('lineup_eastPlayer_firstName')+pl.lit(' ')+pl.col('lineup_eastPlayer_lastName')).alias('Player_Name_E'),
        (pl.col('lineup_southPlayer_firstName')+pl.lit(' ')+pl.col('lineup_southPlayer_lastName')).alias('Player_Name_S'),
        (pl.col('lineup_westPlayer_firstName')+pl.lit(' ')+pl.col('lineup_westPlayer_lastName')).alias('Player_Name_W'),
        pl.col('lineup_northPlayer_id'),
        pl.col('lineup_eastPlayer_id'),
        pl.col('lineup_southPlayer_id'),
        pl.col('lineup_westPlayer_id'),
        pl.col('lineup_northPlayer_ffbId').cast(pl.String).alias('Player_ID_N'),
        pl.col('lineup_eastPlayer_ffbId').cast(pl.String).alias('Player_ID_E'),
        pl.col('lineup_southPlayer_ffbId').cast(pl.String).alias('Player_ID_S'),
        pl.col('lineup_westPlayer_ffbId').cast(pl.String).alias('Player_ID_W'),
        pl.col('lineup_segment_game_homeTeam_id').alias('team_id_home'),
        pl.col('lineup_segment_game_homeTeam_section').alias('section_id_home'),
        pl.col('lineup_segment_game_homeTeam_orientation').alias('Pair_Direction_Home'),
        pl.col('lineup_segment_game_homeTeam_startTableNumber').alias('Pair_Number_Home'),
        pl.col('lineup_segment_game_awayTeam_id').alias('team_id_away'),
        pl.col('lineup_segment_game_awayTeam_section').alias('section_id_away'),
        pl.col('lineup_segment_game_awayTeam_orientation').alias('Pair_Direction_Away'),
        pl.col('lineup_segment_game_awayTeam_startTableNumber').alias('Pair_Number_Away'),
        #pl.col('phase_stade_competitionDivision_competition_label').alias('Game_Description'),
        #pl.col('phase_stade_organization_name').alias('Organization_Name'),
    ])
    assert all(df['section_id_home'] == df['section_id_away'])

    df = df.with_columns([
        pl.col('section_id_home').alias('section_name'),
        pl.col('Score_Freq_List').list.sum().sub(1).alias('MP_Top'), # todo: isn't there a 'top' available?
    ])

    # https://ffbridge.fr/competitions/results/groups/7878/sessions/183872/pairs/8413302 shows Pair_Direction_Home can be 'NS' or 'EW' or '' (sitout).
    #assert all(df['Pair_Direction_Home'].is_in(['NS',''])), df['Pair_Direction_Home'].value_counts() # '' is sitout
    #assert all(df['Pair_Direction_Away'].is_in(['EW',''])), df['Pair_Direction_Away'].value_counts() # '' is sitout
    df = df.with_columns(
        pl.when(pl.col('Pair_Direction_Home').eq('NS'))
            .then(pl.col('Pair_Number_Home'))
            .otherwise(
                pl.when(pl.col('Pair_Direction_Away').eq('NS'))
                    .then(pl.col('Pair_Number_Away'))
                    .otherwise(None)
            )
            .alias('Pair_Number_NS'),
        pl.when(pl.col('Pair_Direction_Home').eq('EW'))
            .then(pl.col('Pair_Number_Home'))
            .otherwise(
                pl.when(pl.col('Pair_Direction_Away').eq('EW'))
                    .then(pl.col('Pair_Number_Away'))
                    .otherwise(None)
            )
            .alias('Pair_Number_EW'),
        #pl.col('section_id_home')+pl.lit('_')+pl.col('Pair_Direction_Home')+pl.col('Pair_Number_Home').cast(pl.Utf8).alias('Pair_ID_NS'),
        #pl.col('section_id_away')+pl.lit('_')+pl.col('Pair_Direction_Away')+pl.col('Pair_Number_Away').cast(pl.Utf8).alias('Pair_ID_EW'),
    )

    # # Filter to keep only rows with the correct orientation
    # df = df.filter(
    #     (pl.col('Pair_Direction_Home').eq('NS')) & 
    #     (pl.col('Pair_Direction_Away').eq('EW'))
    # )
    # # After filtering, you can simplify your column assignments
    # df = df.with_columns([
    #     pl.col('Pair_Number_Home').alias('Pair_Number_NS'),  # Now safe because all home are NS
    #     pl.col('Pair_Number_Away').alias('Pair_Number_EW')   # Now safe because all away are EW
    # ])

    # fails because some boards are sitout(?).
    #assert df['Pair_Number_NS'].is_not_null().all()
    #assert df['Pair_Number_EW'].is_not_null().all()

    df = df.with_columns(
        pl.struct(['Scores_List_NS', 'Scores_List_EW', 'Score_Freq_List'])
            # substitute None for adjusted scores (begin with %).
            .map_elements(lambda x: [None if '%' in score_ns or '%' in score_ew else 0 if score_ns == 'PASS' or score_ew == 'PASS' else int(score_ns) if len(score_ns) else int('-'+score_ew) for score_ns, score_ew, freq in zip(x['Scores_List_NS'], x['Scores_List_EW'], x['Score_Freq_List']) for _ in range(freq)],return_dtype=pl.List(pl.Int16))
            .alias('Expanded_Scores_List')
    )
    df = df.with_columns(
        (pl.col('Pct_NS')*pl.col('MP_Top')).round(2).alias('MP_NS'),
        (pl.col('Pct_EW')*pl.col('MP_Top')).round(2).alias('MP_EW'),
    )

    return df
