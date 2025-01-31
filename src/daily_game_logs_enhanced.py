import pandas as pd
import numpy as np
import xlwings as xw
import os


def calculate_streaks(df, stat, avg_col):
    # Create a boolean mask for above average performances
    above_avg = df[stat] > df[avg_col]

    # Create groups when the streak changes
    streak_groups = above_avg.ne(above_avg.shift()).cumsum()

    # Calculate the streak lengths
    streak_lengths = above_avg.astype(int).groupby(streak_groups).cumsum()

    # Reset streaks to 0 where performance was below average
    return streak_lengths.where(above_avg, 0)



def format_decimals(df, decimal_places=3):
    """Format numeric columns to specified decimal places"""
    for column in df.select_dtypes(include=['float64', 'float32']).columns:
        df[column] = df[column].round(decimal_places)
    return df


def enhance_game_logs(df):
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



def main():
    try:
        import traceback

        # Define file paths
        input_path = os.path.join('..', 'dk_import', 'nba_boxscores.csv')
        output_csv_path = os.path.join('..', 'dk_import', 'nba_boxscores_enhanced.csv')
        excel_path = os.path.join('..', 'dk_import', 'nba_wip.xlsm')

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
        enhanced_df = enhance_game_logs(df)

        # Check for any remaining empty numeric values
        numeric_columns = enhanced_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        empty_counts = enhanced_df[numeric_columns].isna().sum()
        if empty_counts.sum() > 0:
            print("\nWarning: Found empty numeric values in the following columns:")
            for col in empty_counts[empty_counts > 0].index:
                print(f"- {col}: {empty_counts[col]} empty values")

        # Round all decimal places
        enhanced_df = format_decimals(enhanced_df, decimal_places=3)

        # Save enhanced dataset to CSV
        print("Saving enhanced dataset...")
        enhanced_df.to_csv(output_csv_path,
                           index=False,
                           encoding='utf-8',
                           float_format='%.3f')

        #Update existing Excel workbook
        print("Updating Excel workbook...")
        print("  • Getting most recent game date...")
        try:
            # Get the most recent game date from the DataFrame
            most_recent_date = enhanced_df['GAME DATE'].max()

            print("  • Filtering data for last 7 days...")
            # Filter DataFrame for last 7 days from most recent game
            seven_days_ago = most_recent_date - pd.Timedelta(days=7)
            filtered_df = enhanced_df[enhanced_df['GAME DATE'] >= seven_days_ago]
            print(f"    - Found {len(filtered_df)} rows of recent data")

            print("  • Opening Excel application...")
            app = xw.App(visible=False)
            app.display_alerts = False
            app.screen_updating = False

            print("  • Opening workbook...")
            wb = app.books.open(excel_path)
            sheet = wb.sheets['game_logs']
            table = sheet.tables['game_logs']

            print("  • Preparing to update data...")
            # Get the existing table
            data_range = table.data_body_range

            # Calculate the range up to column DB
            last_data_column = 'CU'  # This is the last column before formulas
            if data_range is not None:
                start_row = data_range.row
                end_row = start_row + len(filtered_df) - 1
                limited_range = sheet.range(f"A{start_row}:{last_data_column}{end_row}")

                print("  • Clearing existing data...")
                limited_range.clear_contents()

                print("  • Writing new data...")
                # Convert DataFrame to values and get total rows
                data_values = filtered_df.values
                total_rows = len(data_values)

                # Write data in chunks and show progress
                chunk_size = max(1, total_rows // 10)  # Show progress every 5%
                for i in range(0, total_rows, chunk_size):
                    # Calculate current chunk
                    end_idx = min(i + chunk_size, total_rows)
                    chunk_range = sheet.range(f"A{start_row + i}:{last_data_column}{start_row + end_idx - 1}")

                    # Write chunk
                    chunk_range.value = data_values[i:end_idx]

                    # Calculate and display progress
                    progress = min(100, (end_idx / total_rows) * 100)
                    print(f"    - Progress: {progress:.1f}% ({end_idx}/{total_rows} rows)")

                print("  • Saving workbook...")
                wb.save()

                print("  • Cleaning up...")
                wb.close()
                app.quit()

                print("✓ Excel table updated successfully!")
            else:
                print("⚠ Warning: No data range found in Excel table")

        except Exception as e:
            print(f"❌ Error updating Excel: {str(e)}")
            print("Full error details:")
            traceback.print_exc()

            # Ensure Excel is properly closed even if there's an error
            try:
                wb.close()
                app.quit()
            except:
                pass



        print("Processing completed successfully!")
        print(f"Total rows processed: {len(enhanced_df)}")
        print(f"Total columns created: {len(enhanced_df.columns)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Full error details:")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()