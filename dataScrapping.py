############# all imports ################
# datasets
import pandas as pd
import nfl_data_py as nfl
import numpy
# ML Models
import torch

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

def generateData(inputStats, outputStats, yearRange, position):
    # import draft data from years specified
    d = nfl.import_draft_picks(yearRange)
    df1 = pd.DataFrame(d)

    # decide what input and output stats we want for our model
    df1 = df1[["pfr_player_id"] + inputStats]

    # filter out all rows to only include running backs
    draft_rbs = df1[df1['position'] == position]

    #rename column for consistency with combine dataframe later
    draft_rbs.rename(columns = {"pfr_player_id":"pfr_id"}, inplace = True)

    #remove players who don't have a pfr_id
    draft_rbs = draft_rbs[draft_rbs['pfr_id'].notna()]

    # import combine data from years specified
    d = nfl.import_combine_data(yearRange, [position])
    df2 = pd.DataFrame(d)

    # decide what input stats we want from the combine
    combine_rbs = df2[["pfr_id"] + outputStats]

    # new dataframe with added stats for each drafted running back
    rb_stats = appendIndStats(draft_rbs, combine_rbs, "pfr_id", outputStats)
    
    # fill all NaNs with zeros
    rb_stats.fillna(0, inplace=True)
    
    # convert height values from feet-inches to inches
    rb_stats['ht'] = rb_stats['ht'].apply(heightConversion)
    
    return rb_stats

inputs = ["pick", "pfr_player_name", "position", "rush_atts", "rush_yards", "rush_tds", "receptions", "rec_yards", "rec_tds", "games"]
outputs = ["ht", "wt", "forty", "bench", "vertical", "broad_jump"]
years = [2017]
rb_stats = generateData(inputs, outputs, years, 'RB')
print(rb_stats)