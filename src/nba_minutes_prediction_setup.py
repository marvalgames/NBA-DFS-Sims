import pandas as pd
import numpy as np
import xlwings as xw
import pandas as pd
import numpy as np
import xlwings as xw
import os
import time
import unicodedata
import pandas as pd

import os
class NbaMInutesPredictions:

    def __init__(self):
        pass

    def calculate_streaks(self, df, stat, avg_col):
        # Create a boolean mask for above average performances
        above_avg = df[stat] > df[avg_col]

        # Create groups when the streak changes
        streak_groups = above_avg.ne(above_avg.shift()).cumsum()

        # Calculate the streak lengths
        streak_lengths = above_avg.astype(int).groupby(streak_groups).cumsum()

        # Reset streaks to 0 where performance was below average
        return streak_lengths.where(above_avg, 0)



    def format_decimals(self, df, decimal_places=3):
        """Format numeric columns to specified decimal places"""
        for column in df.select_dtypes(include=['float64', 'float32']).columns:
            df[column] = df[column].round(decimal_places)
        return df

    def nba_enhance_game_logs(self):
        try:
            import traceback

            # Define file paths
            input_path = os.path.join('..', 'dk_import', 'nba_boxscores.csv')
            output_csv_path = os.path.join('..', 'dk_import', 'nba_boxscores_enhanced.csv')

            # Read the CSV file with different encoding
            print("Reading input file...")
            try:
                df = pd.read_csv(input_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(input_path, encoding='cp1252')
                except UnicodeDecodeError:
                    df = pd.read_csv(input_path, encoding='latin-1')

            print(f"Successfully read {len(df)} rows")

            # Process the data
            print("Enhancing dataset...")
            enhanced_df = predictions.enhance_game_logs(df)

            # Check for any remaining empty numeric values
            numeric_columns = enhanced_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            empty_counts = enhanced_df[numeric_columns].isna().sum()
            if empty_counts.sum() > 0:
                print("\nWarning: Found empty numeric values in the following columns:")
                for col in empty_counts[empty_counts > 0].index:
                    print(f"- {col}: {empty_counts[col]} empty values")

            # Round all decimal places
            enhanced_df = predictions.format_decimals(enhanced_df, decimal_places=3)

            # Save enhanced dataset to CSV
            print("Saving enhanced dataset...")
            enhanced_df.to_csv(output_csv_path,
                               index=False,
                               encoding='utf-8',
                               float_format='%.3f')

            print("Processing completed successfully!")
            print(f"Total rows processed: {len(enhanced_df)}")
            print(f"Total columns created: {len(enhanced_df.columns)}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Full error details:")
            import traceback
            print(traceback.format_exc())



    def enhance_game_logs(self, df):
        # Initialize dictionary for new columns
        new_columns = {}

        # Initial datetime conversion
        df['GAME DATE'] = pd.to_datetime(df['GAME_DATE'])

        # Create game day counter
        unique_dates = df['GAME DATE'].unique()
        date_to_gameday = {date: idx + 1 for idx, date in enumerate(sorted(unique_dates))}

        # Create temporary DataFrame and add GAME_DAY to it
        temp_df = df.copy()
        temp_df['GAME_DAY'] = temp_df['GAME DATE'].map(date_to_gameday)
        new_columns['GAME_DAY'] = temp_df['GAME_DAY']

        # Fill numeric columns with 0 for empty values
        numeric_columns = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Basic column mappings - add these directly to both DataFrames
        df['PLAYER'] = df['PLAYER_NAME']
        df['DK'] = df['FANTASY_PTS']
        df['PLUS MINUS'] = df['PLUS_MINUS']
        df['MATCH UP'] = df['MATCHUP']
        df['W/L'] = df['WL']
        df['TEAM'] = df['TEAM_NAME']

        temp_df['PLAYER'] = df['PLAYER_NAME']
        temp_df['DK'] = df['FANTASY_PTS']
        temp_df['PLUS MINUS'] = df['PLUS_MINUS']
        temp_df['MATCH UP'] = df['MATCHUP']
        temp_df['W/L'] = df['WL']
        temp_df['TEAM'] = df['TEAM_NAME']

        # Sort DataFrames
        df = df.sort_values(['PLAYER', 'GAME DATE'])
        temp_df = temp_df.sort_values(['PLAYER', 'GAME DATE'])

        # Create grouped objects for reuse
        grouped_player = df.groupby('PLAYER')
        grouped_team_date = df.groupby(['TEAM', 'GAME DATE'])
        temp_grouped_player = temp_df.groupby('PLAYER')

        # Basic stats calculations
        stats = ['MIN', 'PTS', 'REB', 'AST', 'DK', 'PLUS MINUS']

        for stat in stats:
            # Calculate stats in temp_df first
            temp_df[f'{stat}_LAST_10_AVG'] = temp_grouped_player[stat].transform(
                lambda x: x.rolling(10, min_periods=1).mean())
            temp_df[f'{stat}_LAST_5_AVG'] = temp_grouped_player[stat].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
            temp_df[f'{stat}_LAST_3_AVG'] = temp_grouped_player[stat].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            temp_df[f'{stat}_CUM_AVG'] = temp_grouped_player[stat].transform(
                lambda x: x.expanding().mean())

            # Store in new_columns
            new_columns[f'{stat}_LAST_10_AVG'] = temp_df[f'{stat}_LAST_10_AVG']
            new_columns[f'{stat}_LAST_5_AVG'] = temp_df[f'{stat}_LAST_5_AVG']
            new_columns[f'{stat}_LAST_3_AVG'] = temp_df[f'{stat}_LAST_3_AVG']
            new_columns[f'{stat}_CUM_AVG'] = temp_df[f'{stat}_CUM_AVG']

            # Performance trend
            rolling_5 = temp_df[f'{stat}_LAST_5_AVG']
            new_columns[f'{stat}_TREND'] = (rolling_5 > temp_df[f'{stat}_CUM_AVG']).astype(int)

        # Days between games (rest days)
        new_columns['DAYS_REST'] = grouped_player['GAME DATE'].transform(
            lambda x: (x - x.shift(1)).dt.days)

        # Game situation features
        new_columns['IS_B2B'] = (new_columns['DAYS_REST'] == 1).astype(int)
        new_columns['IS_HOME'] = df['MATCH UP'].apply(lambda x: 0 if '@' in x else 1)
        new_columns['WIN'] = (df['W/L'] == 'W').astype(int)

        # Efficiency metrics
        temp_df['PTS_PER_MIN'] = df['PTS'] / df['MIN']
        temp_df['AST_PER_MIN'] = df['AST'] / df['MIN']
        temp_df['REB_PER_MIN'] = df['REB'] / df['MIN']

        new_columns['PTS_PER_MIN'] = temp_df['PTS_PER_MIN']
        new_columns['AST_PER_MIN'] = temp_df['AST_PER_MIN']
        new_columns['REB_PER_MIN'] = temp_df['REB_PER_MIN']

        # Performance vs team average
        team_stats = ['MIN', 'PTS', 'REB', 'AST']
        for stat in team_stats:
            new_columns[f'{stat}_VS_TEAM_AVG'] = grouped_team_date[stat].transform(
                lambda x: x / x.mean())

        # Consistency metrics
        for stat in stats:
            new_columns[f'{stat}_CONSISTENCY'] = grouped_player[stat].transform(
                lambda x: x.rolling(10, min_periods=1).std())

        # Fantasy performance trends
        new_columns['DK_TREND_5'] = grouped_player['DK'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().diff())

        # Game margin categories
        new_columns['BLOWOUT_GAME'] = (abs(df['PLUS MINUS']) > 20).astype(int)

        # Minutes Distribution Features
        for threshold in [20, 25, 30]:
            col_name = f'MIN_ABOVE_{threshold}'
            # First create the threshold columns in the temp DataFrame
            temp_df[col_name] = (df['MIN'] > threshold).astype(int)
            # Now we can use groupby on these columns
            new_columns[f'FREQ_ABOVE_{threshold}'] = temp_df.groupby('PLAYER')[col_name].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )
            # Also store the binary threshold columns in new_columns
            new_columns[col_name] = temp_df[col_name]

            # Streak indicators (new version)
        for stat in ['MIN', 'PTS']:
            # Process each player separately to avoid the groupby.apply
            players = temp_df['PLAYER'].unique()
            streaks = pd.Series(index=temp_df.index, dtype=float)

            for player in players:
                player_mask = temp_df['PLAYER'] == player
                player_data = temp_df[player_mask].copy()
                player_streaks = self.calculate_streaks(
                    player_data,
                    stat,
                    f'{stat}_CUM_AVG'
                )
                streaks[player_mask] = player_streaks
            new_columns[f'{stat}_ABOVE_AVG_STREAK'] = streaks

        # Team minutes and rank features - using temp_df for calculations
        new_columns['TEAM_PROJ_RANK'] = temp_df.groupby(['TEAM', 'GAME_DAY'])['DK_LAST_10_AVG'].rank(ascending=False)
        new_columns['IS_TOP_3_PROJ'] = (new_columns['TEAM_PROJ_RANK'] <= 3).astype(int)

        team_totals = grouped_team_date['MIN'].transform('sum')
        new_columns['TEAM_MIN_PERCENTAGE'] = (df['MIN'] / team_totals) * 100

        new_columns['LOW_MIN_TOP_PLAYER'] = ((new_columns['TEAM_PROJ_RANK'] <= 3) &
                                             (new_columns['TEAM_MIN_PERCENTAGE'] < 14)).astype(int)

        # Role change and consistency metrics
        new_columns['ROLE_CHANGE_3_10'] = (temp_df['MIN_LAST_3_AVG'] -
                                           temp_df['MIN_LAST_10_AVG']).abs()
        new_columns['ROLE_CHANGE_5_10'] = (temp_df['MIN_LAST_5_AVG'] -
                                           temp_df['MIN_LAST_10_AVG']).abs()

        new_columns['MIN_CONSISTENCY_SCORE'] = grouped_player['MIN'].transform(
            lambda x: x.rolling(10, min_periods=1).std() / x.rolling(10, min_periods=1).mean())

        # Performance and impact features - UPDATED
        # First add these columns to temp_df
        temp_df['SCORING_EFFICIENCY'] = df['PTS'] / df['MIN']
        temp_df['PLUS_MINUS_PER_MIN'] = df['PLUS MINUS'] / df['MIN']

        # Now we can use them in groupby operations
        new_columns['SCORING_EFFICIENCY'] = temp_df['SCORING_EFFICIENCY']
        new_columns['RECENT_SCORING_EFF'] = temp_df.groupby('PLAYER')['SCORING_EFFICIENCY'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())

        new_columns['PLUS_MINUS_PER_MIN'] = temp_df['PLUS_MINUS_PER_MIN']
        new_columns['RECENT_IMPACT'] = temp_df.groupby('PLAYER')['PLUS_MINUS_PER_MIN'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())





        # Combine all new columns with original DataFrame, but sort only the new columns first
        original_cols = df.columns.tolist()
        new_cols_sorted = sorted(new_columns.keys())

        # Create DataFrame with sorted new columns
        new_df = pd.DataFrame(new_columns, columns=new_cols_sorted)

        # Combine with original DataFrame, maintaining original column order
        enhanced_df = pd.concat([df, new_df], axis=1)

        # Final cleanup of any remaining NaN values
        numeric_columns = enhanced_df.select_dtypes(
            include=['float64', 'float32', 'int64', 'int32']).columns
        enhanced_df[numeric_columns] = enhanced_df[numeric_columns].fillna(0)

        return enhanced_df

    def process_excel_data(self):
        # Read from Excel using XWings

        app = xw.App(visible=False)
        time.sleep(1)  # Give Excel a moment to fully initialize
        # Connect to Excel
        # excel_path = os.path.join('..', 'dk_import', 'nba - Copy.xlsm')
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
        result_df = self.merge_latest_records_with_columns(excel_df, combined_df, excel_key='Player',
                                                      combined_key='PLAYER_NAME', date_column="GAME_DATE")
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

    def standardize_player_names(self, df, name_column):
        """
        Standardizes player names according to specific rules
        """
        if not isinstance(df, pd.DataFrame):
            return df

        # Create a copy to avoid modifying the original
        df = df.copy()

        name_mappings = {
            "Herb Jones": "Herbert Jones",
            "GG Jackson": "Gregory Jackson",
            "G.G. Jackson": "Gregory Jackson",
            "Alexandre Sarr": "Alex Sarr",
            "Yongxi Cui": "Cui Yongxi",
            "Nicolas Claxton": "Nic Claxton",
            "Cameron Johnson": "Cam Johnson",
            "Kenyon Martin Jr": "KJ Martin",
            "Ronald Holland": "Ron Holland",
            "Nah'Shon Hyland": "Bones Hyland",
            "Elijah Harkless": "EJ Harkless",
            "Cameron Payne": "Cam Payne",
            "Bub Carrington": "Carlton Carrington",
            "Jabari Smith Jr": "Jabari Smith",
            "Gary Trent Jr": "Gary Trent",
            "Tim Hardaway Jr": "Tim Hardaway",
            "Michael Porter Jr": "Michael Porter",
            "Kelly Oubre Jr": "Kelly Oubre",
            "Patrick Baldwin Jr": "Patrick Baldwin",
            "Kevin Knox II": "Kevin Knox",
            "Trey Murphy III": "Trey Murphy",
            "Wendell Moore Jr": "Wendell Moore",
            "Vernon Carey Jr": "Vernon Carey",
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
            lambda x: x.replace(" Jr", "").replace("III", "").replace("II", "").replace("'", "").replace(".",
                                                                                                         "").strip()
            if isinstance(x, str) else x
        )

        return df

    def merge_latest_records_with_columns(self, excel_df, combined_df,
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
        excel_df =self.standardize_player_names(excel_df, excel_key)
        combined_df = self.standardize_player_names(combined_df, combined_key)

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


#def main():
if __name__ == "__main__":

    predictions = NbaMInutesPredictions()
    predictions.nba_enhance_game_logs()
    predictions.process_excel_data()
    print('Completed')

#    if __name__ == "__main__":
#        main()