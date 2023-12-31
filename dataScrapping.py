############# all imports ################
# datasets
import numpy as np
import pandas as pd
import nfl_data_py as nfl

##########################################

# ["season", "pfr_player_name", "pfr_player_id", "pick", "team", "position", "rush_atts", "rush_yards", "rush_tds", "receptions", "rec_yards", "rec_tds", "category", "games", "seasons_started"]
# ["pfr_id", "draft_team", "player_name", "pos", "school", "ht", "wt", "forty", "bench", "vertical", "broad_jump"]

def heightConversion(height_str):
    if isinstance(height_str, int):
        return height_str
    
    feet, inches = map(int , height_str.split('-'))
    return (feet * 12) + inches

def appendIndStats(source, target, columnOn, columnsToBeAppended):
    merged_df = pd.merge(source, target[columnsToBeAppended + [columnOn]], on = columnOn, how = 'left', suffixes = ('_df1', '_df2'))
    # merged_df = merged_df.drop(columnOn, axis = 1)
    return merged_df

def generateData(inputStats, outputStats, yearRange, position, OmitNoShows=False):
    # import draft data from years specified
    d = nfl.import_draft_picks(yearRange)
    df1 = pd.DataFrame(d)

    # decide what input and output stats we want for our model
    df1 = df1[["pfr_player_id", "pick", "pfr_player_name", "position"] + outputStats]

    # filter out all rows to only include running backs
    drafts = df1[df1['position'] == position]

    # rename column for consistency with combine dataframe later
    drafts.rename(columns = {"pfr_player_id":"pfr_id"}, inplace = True)

    # remove players who don't have a pfr_id
    drafts = drafts[drafts['pfr_id'].notna()]

    # import combine data from years specified
    d = nfl.import_combine_data(yearRange, [position])
    df2 = pd.DataFrame(d)
    print(df2.columns)
    # decide what input stats we want from the combine
    combines = df2[["pfr_id"] + inputStats]

    # new dataframe with added stats for each drafted running back
    pos_stats = appendIndStats(drafts, combines, "pfr_id", inputStats)
    
    # fill all NaNs with zeros
    pos_stats.fillna(0, inplace=True)
    
    # convert height values from feet-inches to inches
    pos_stats['ht'] = pos_stats['ht'].apply(heightConversion)
    
    # omit noshows to combine if option is enabled
    if OmitNoShows:
        pos_stats = pos_stats[pos_stats['vertical'] != 0.0]
    return pos_stats
