
import bridgestats


if __name__ == '__main__':
    chart_options = ['Declarer_ParScore','ContractType','Declarer_ParScore','Declarer_Pct','Declarer_DD_Tricks','Declarer_DD_Score','Declarer_DD_Pct','Declarer_Tricks_DD_Diff','Declarer_Score_DD_Diff','Declarer_ParScore_DD_Diff']
    club_or_tournament = 'tournament'
    pair_or_player = 'player'
    groupby=['Declarer','Dummy']
    bridgestats.Stats(club_or_tournament, pair_or_player, chart_options, groupby)
