

import pandas as pd
import numpy as np
import xlwings as xw
import os
import time


def process_excel_data():
    # Read from Excel using XWings

    app = xw.App(visible=False)
    time.sleep(1)  # Give Excel a moment to fully initialize
    # Connect to Excel
    #excel_path = os.path.join('..', 'dk_import', 'nba - Copy.xlsm')
    excel_path = os.path.join('..', 'dk_import', 'nba.xlsm')
    csv_path = os.path.join('..', 'dk_import', 'nba_daily_combined.csv')
    csv_read = os.path.join('..', 'dk_import', 'nba_boxscores_enhanced.csv')

    wb = app.books.open(excel_path)
    ws = wb.sheets['sog_minutes']
    table = ws.tables['sog_minutes']
    data_range = table.data_body_range
    last_row = ws.range('A' + str(ws.cells.last_cell.row)).end('up').row

    # Read data
    needed_data = {
        'Player': ws.range(f'A2:A{last_row}').value,
        'Team': ws.range(f'B2:B{last_row}').value,
        'Position': ws.range(f'C2:C{last_row}').value,
        'Salary': ws.range(f'D2:D{last_row}').value,
        'Minutes': ws.range(f'E2:E{last_row}').value,
        'Max Minutes': ws.range(f'J2:J{last_row}').value,
        'Projection': ws.range(f'M2:M{last_row}').value,

    }

    excel_df = pd.DataFrame(needed_data)


    # Load your combined DataFrame
    combined_df = pd.read_csv(csv_read, encoding='utf-8')  # or however you load it
    result_df = merge_latest_records_with_columns(excel_df, combined_df, excel_key='Player', combined_key='PLAYER_NAME', date_column="GAME_DATE" )
    result_df = result_df.rename(columns={'Projection': 'Projected Pts'})
    # Define the new column order
    new_column_order = [
        # Core Player Info
        'Player', 'Team', 'Position', 'Salary',

        # Minutes Data
        'Minutes', 'Max Minutes', 'MIN_CUM_AVG', 'MIN_LAST_3_AVG', 'MIN_LAST_5_AVG', 'MIN_LAST_10_AVG',
        'MIN_TREND', 'MIN_CONSISTENCY', 'MIN_CONSISTENCY_SCORE',
        'MIN_ABOVE_20', 'MIN_ABOVE_25', 'MIN_ABOVE_30',
        'MIN_ABOVE_AVG_STREAK',

        'FREQ_ABOVE_20',
        'FREQ_ABOVE_25',
        'FREQ_ABOVE_30',

        # DraftKings Scoring
        'Projected Pts', 'DK', 'DK_CUM_AVG', 'DK_LAST_3_AVG', 'DK_LAST_5_AVG', 'DK_LAST_10_AVG',
        'DK_TREND', 'DK_TREND_5', 'DK_CONSISTENCY',

        # Key Stats - Recent Averages
        'PTS_LAST_3_AVG', 'PTS_LAST_5_AVG', 'PTS_LAST_10_AVG',
        'REB_LAST_3_AVG', 'REB_LAST_5_AVG', 'REB_LAST_10_AVG',
        'AST_LAST_3_AVG', 'AST_LAST_5_AVG', 'AST_LAST_10_AVG',

        # Cumulative Averages
        'PTS_CUM_AVG', 'REB_CUM_AVG', 'AST_CUM_AVG',

        # Efficiency Metrics
        'PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN',
        'SCORING_EFFICIENCY', 'RECENT_SCORING_EFF',

        # Shooting Stats
        'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',

        # Game Impact Stats
        'PLUS_MINUS', 'PLUS_MINUS_PER_MIN',
        'PLUS MINUS_LAST_3_AVG', 'PLUS MINUS_LAST_5_AVG', 'PLUS MINUS_LAST_10_AVG',
        'PLUS MINUS_CUM_AVG', 'PLUS MINUS_TREND', 'PLUS MINUS_CONSISTENCY',

        # Trend Analysis
        'PTS_TREND', 'REB_TREND', 'AST_TREND',
        'PTS_CONSISTENCY', 'REB_CONSISTENCY', 'AST_CONSISTENCY',

        # Team Context
        'TEAM_MIN_PERCENTAGE', 'TEAM_PROJ_RANK',
        'PTS_VS_TEAM_AVG', 'REB_VS_TEAM_AVG', 'AST_VS_TEAM_AVG', 'MIN_VS_TEAM_AVG',

        # Role Analysis
        'ROLE_CHANGE_3_10', 'ROLE_CHANGE_5_10',
        'IS_TOP_3_PROJ', 'LOW_MIN_TOP_PLAYER',

        # Game Info
        'GAME_DATE', 'IS_HOME', 'IS_B2B', 'DAYS_REST',
        'MATCHUP', 'WL', 'BLOWOUT_GAME',

        # Additional Stats
        'STL', 'BLK', 'TOV', 'OREB', 'DREB', 'PF',

        # Metadata
        'PLAYER_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'SEASON_ID',
        'VIDEO_AVAILABLE'


    ]

    # Reorder the DataFrame
    result_df = result_df[new_column_order]

    # Create two DataFrames: one for columns 1-6 and one for columns 7 onwards
    df_first_part = result_df.iloc[:, :6]  # First 6 columns
    df_to_write = result_df.iloc[:, 6:]  # From 7th column onwards

    excel_start_row = 2
    excel_start_col = 19  # Starting column in Excel

    # Clear the destination range first
    clear_range = ws.range(
        (excel_start_row, excel_start_col),
        (excel_start_row + 650, excel_start_col + len(result_df.columns) - 1)
    )
    if clear_range is None:
        raise ValueError("Failed to create clear range")
    clear_range.clear_contents()

    # Create the write range starting from the specified Excel column
    write_range = ws.range(
        (excel_start_row, excel_start_col),
        (excel_start_row + len(df_to_write.index) - 1, excel_start_col + len(df_to_write.columns) - 1)
    )

    # Write the DataFrame values to the specified range
    write_range.value = df_to_write.values

    print("Saving workbook...")
    wb.save()
    print("Import completed successfully.")

    wb.close()
    app.quit()
    result_df.to_csv(csv_path, index=False)
    return result_df


import unicodedata
import pandas as pd


def standardize_player_names(df, name_column):
    """
    Standardizes player names according to specific rules
    """
    if not isinstance(df, pd.DataFrame):
        return df

    # Create a copy to avoid modifying the original
    df = df.copy()

    name_mappings = {
        # ... your existing mappings ...
        "Kristaps PorziÅ†Ä£is": "Kristaps Porzingis"  # Add this specific mapping if needed
    }

    def remove_accents(text):
        if not isinstance(text, str):
            return text
        try:
            # Normalize to decomposed form (separate letters from accents)
            nfkd_form = unicodedata.normalize('NFKD', text)
            # Remove non-ASCII characters (like accents)
            return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        except:
            # Fallback method if Unicode normalization fails
            accents = "áàâãäéèêëíìîïóòôõöúùûüýÿÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÝćčćdšžCCÐŠŽūÅ†Ä£"
            regular = "aaaaaeeeeiiiiooooouuuuyyAAAAAEEEEIIIIOOOOOUUUUYcccdszCCDSZuAng"

            for accent, reg in zip(accents, regular):
                text = text.replace(accent, reg)
            return text

    # Apply standardization to the name column
    df[name_column] = df[name_column].apply(lambda x: x if not isinstance(x, str) else x.strip())

    # Apply name mappings
    for old_name, new_name in name_mappings.items():
        df[name_column] = df[name_column].apply(
            lambda x: new_name if isinstance(x, str) and x.strip() == old_name else x
        )

    # Remove accents and special characters
    df[name_column] = df[name_column].apply(
        lambda x: remove_accents(x) if isinstance(x, str) else x
    )

    # Remove suffixes and special characters
    df[name_column] = df[name_column].apply(
        lambda x: x.replace(" Jr", "").replace("III", "").replace("II", "").replace("'", "").replace(".", "").strip()
        if isinstance(x, str) else x
    )

    return df


def merge_latest_records_with_columns(excel_df, combined_df,
                                      excel_key='Player',
                                      combined_key='PLAYER_NAME',
                                      date_column='Game_Date',
                                      columns_to_merge=None):
    """
    Merges DataFrames with different key column names and writes to both Excel and CSV
    """
    print("Starting merge process...")
    print(f"Excel DataFrame contains {len(excel_df)} rows")
    print(f"Combined DataFrame contains {len(combined_df)} rows")

    # First, identify rows with actual players
    excel_df['has_player'] = excel_df[excel_key].notna() & (excel_df[excel_key] != '')

    print("\nStandardizing player names...")
    excel_df = standardize_player_names(excel_df, excel_key)
    combined_df = standardize_player_names(combined_df, combined_key)



    print("Creating working copy of data...")
    working_df = combined_df.copy()

    # Convert date column to datetime
    working_df[date_column] = pd.to_datetime(working_df[date_column])

    # Rename the key column to match excel_df
    working_df = working_df.rename(columns={combined_key: excel_key})

    # If no columns specified, use all columns from working_df
    if columns_to_merge is None:
        columns_to_merge = working_df.columns.tolist()

    # Ensure key and date columns are included
    required_columns = [excel_key, date_column]
    columns_to_use = list(set(required_columns + columns_to_merge))

    # Filter and get latest records
    print("\nFiltering for matching players...")
    filtered_df = working_df[working_df[excel_key].isin(excel_df[excel_key])]
    print(f"Found {len(filtered_df)} matching records")

    print("Getting latest records for each player...")
    latest_records = (filtered_df[columns_to_use]
                      .sort_values(date_column)
                      .groupby(excel_key)
                      .last()
                      .reset_index())

    print(f"Retrieved {len(latest_records)} latest records")

    # Drop date column if it wasn't in columns_to_merge
    if date_column not in columns_to_merge:
        latest_records = latest_records.drop(columns=[date_column])

    # Merge with excel_df
    result_df = excel_df.merge(latest_records,
                               on=excel_key,
                               how='left')

    # Handle missing values for player rows only
    print("Filling missing values with zeros for player rows only...")
    # Get numeric columns only
    numeric_columns = result_df.select_dtypes(include=['int64', 'float64']).columns
    columns_to_fill = [col for col in numeric_columns
                       if col != excel_key and col != 'has_player']

    # Create a mask for player rows
    player_mask = result_df['has_player']

    # Fill values for numeric columns only
    for col in columns_to_fill:
        null_mask = result_df[col].isna()
        if null_mask.any():
            result_df.loc[player_mask & null_mask, col] = 0

    # Drop the helper column
    result_df = result_df.drop(columns=['has_player'])

    print(f"\nMerge complete! Final DataFrame contains {len(result_df)} rows")

    # Print summary of matches/non-matches
    matched_players = result_df[result_df[columns_to_fill[0]] != 0].shape[0]
    print(f"Players with matching data: {matched_players}")
    print(f"Players without matching data: {len(result_df) - matched_players}")

    return result_df


if __name__ == "__main__":
    process_excel_data()