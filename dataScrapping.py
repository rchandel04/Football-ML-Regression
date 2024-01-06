############# all imports ################
# datasets
import numpy as np
import pandas as pd
import nfl_data_py as nfl

##########################################

# ["season", "pfr_player_name", "pfr_player_id", "pick", "team", "position", "rush_atts", "rush_yards", "rush_tds", "receptions", "rec_yards", "rec_tds", "category", "games", "seasons_started"]
# ["pfr_id", "draft_team", "player_name", "pos", "school", "ht", "wt", "forty", "bench", "vertical", "broad_jump"]

def height_conversion(height_str):
    if isinstance(height_str, int):
        return height_str
    
    feet, inches = map(int , height_str.split('-'))
    return (feet * 12) + inches

def append_ind_stats(source, target, column_on, columns_to_be_appended):
    merged_df = pd.merge(source, target[columns_to_be_appended + [column_on]], on = column_on, how = 'left', suffixes = ('_df1', '_df2'))
    # merged_df = merged_df.drop(columnOn, axis = 1)
    return merged_df

def generate_data(input_stats, output_stats, year_range, position, omit_no_shows=False):
    # import draft data from years specified
    d = nfl.import_draft_picks(year_range)
    df1 = pd.DataFrame(d)

    # decide what input and output stats we want for our model
    df1 = df1[["pfr_player_id", "pick", "pfr_player_name", "position"] + output_stats]

    # filter out all rows to only include running backs
    drafts = df1[df1['position'] == position]

    # rename column for consistency with combine dataframe later
    drafts.rename(columns = {"pfr_player_id":"pfr_id"}, inplace = True)

    # remove players who don't have a pfr_id
    drafts = drafts[drafts['pfr_id'].notna()]

    # import combine data from years specified
    d = nfl.import_combine_data(year_range, [position])
    df2 = pd.DataFrame(d)
    print(df2.columns)
    # decide what input stats we want from the combine
    combines = df2[["pfr_id"] + input_stats]

    # new dataframe with added stats for each drafted running back
    pos_stats = append_ind_stats(drafts, combines, "pfr_id", input_stats)
    
    # fill all NaNs with zeros
    pos_stats.fillna(0, inplace=True)
    
    # convert height values from feet-inches to inches
    pos_stats['ht'] = pos_stats['ht'].apply(height_conversion)
    
    # omit noshows to combine if option is enabled
    if omit_no_shows:
        pos_stats = pos_stats[pos_stats['vertical'] != 0.0]
    return pos_stats
