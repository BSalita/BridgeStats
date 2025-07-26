# mlBridgeLib package
from mlBridgeLib.mlBridgeLib import (
    pd_options_display,
    Direction_to_NESW_d,
    brs_to_pbn,
    contract_classes,
    strain_classes,
    level_classes,
    dbl_classes,
    direction_classes,
    Vulnerability_to_Vul_d,
    json_to_sql_walk,
    CreateSqlFile,
    NESW,
    SHDC,
    CDHS,
    CDHSN,
    NS_EW,
    NSHDC,
    HandsToBin,
    BoardNumberToVul,
    validate_brs,
    seats,
    vul_syms,
    ranked_suit,
    PlayerDirectionToPairDirection,
    NextPosition,
    PairDirectionToOpponentPairDirection,
    declarer_direction_to_pair_direction,
    BoardNumberToDealer,
    ContractToScores,
    DirectionToVul,
    hands_to_brs,
    pbn_to_hands,
    score,
    ContractTypeFromContract,
    ContractType,
    HandsToHCP,
    HandsToQT,
    HandsToSuitLengths,
    HandsToDistributionPoints,
    LoTT_SHDC,
    CategorifyContractTypeBySuit,
    CategorifyContractTypeByDirection,
    MatchPointScoreUpdate,
)

# List of all possible contract strings
contract_classes = [f"{level}{strain}{dbl}" for level in range(1,8) for strain in ['C','D','H','S','N'] for dbl in ['','X','XX']] + ['Pass']

# List of all possible strains
strain_classes = ['C', 'D', 'H', 'S', 'N']

# List of all possible bid levels
level_classes = list(range(1,8))

# List of all possible double states
dbl_classes = ['', 'X', 'XX']

# List of all possible directions
direction_classes = ['N', 'E', 'S', 'W'] 