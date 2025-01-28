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


def enhance_game_logs(df):
    # Convert GAME DATE to datetime if not already
    df['GAME DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Fill numeric columns with 0 for empty values
    numeric_columns = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Create game day counter
    df = df.sort_values('GAME DATE')
    unique_dates = df['GAME DATE'].unique()
    date_to_gameday = {date: idx + 1 for idx, date in enumerate(sorted(unique_dates))}
    df['GAME_DAY'] = df['GAME DATE'].map(date_to_gameday)
    df['PLAYER'] = df['PLAYER_NAME']
    df['DK'] = df['FANTASY_PTS']
    df['PLUS MINUS'] = df['PLUS_MINUS']
    df['MATCH UP'] = df['MATCHUP']
    df['W/L'] = df['WL']
    df['TEAM'] = df['TEAM_NAME']
    # Sort by player and date for the rest of the calculations
    df = df.sort_values(['PLAYER', 'GAME DATE'])


    # Basic stats to track
    stats = ['MIN', 'PTS', 'REB', 'AST', 'DK', 'PLUS MINUS']

    for stat in stats:
        # Rolling 10-game averages
        df[f'{stat}_LAST_10_AVG'] = df.groupby('PLAYER')[stat].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )

        # Cumulative averages
        df[f'{stat}_CUM_AVG'] = df.groupby('PLAYER')[stat].transform(
            lambda x: x.expanding().mean()
        )

        # Performance trend (positive if last 5 games average > season average)
        df[f'{stat}_TREND'] = (
                df.groupby('PLAYER')[stat].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                ) > df[f'{stat}_CUM_AVG']
        ).astype(int)

    # Days between games (rest days)
    df['DAYS_REST'] = df.groupby('PLAYER')['GAME DATE'].transform(
        lambda x: (x - x.shift(1)).dt.days
    )

    # Back to back indicator (0 days rest)
    df['IS_B2B'] = (df['DAYS_REST'] == 1).astype(int)

    # Home/Away indicator
    df['IS_HOME'] = df['MATCH UP'].apply(lambda x: 0 if '@' in x else 1)

    # Team performance indicators
    df['WIN'] = (df['W/L'] == 'W').astype(int)

    # Efficiency metrics
    df['PTS_PER_MIN'] = df['PTS'] / df['MIN']
    df['AST_PER_MIN'] = df['AST'] / df['MIN']
    df['REB_PER_MIN'] = df['REB'] / df['MIN']

    # Performance vs team average
    team_stats = ['MIN', 'PTS', 'REB', 'AST']
    for stat in team_stats:
        df[f'{stat}_VS_TEAM_AVG'] = df.groupby(['TEAM', 'GAME DATE'])[stat].transform(
            lambda x: x / x.mean()
        )

    # Consistency metrics (standard deviation of last 10 games)
    for stat in stats:
        df[f'{stat}_CONSISTENCY'] = df.groupby('PLAYER')[stat].transform(
            lambda x: x.rolling(10, min_periods=1).std()
        )

    # Streak indicators (new version)
    for stat in ['MIN', 'PTS']:
        # Process each player separately to avoid the groupby.apply
        players = df['PLAYER'].unique()
        streaks = pd.Series(index=df.index, dtype=float)

        for player in players:
            player_mask = df['PLAYER'] == player
            player_data = df[player_mask].copy()
            player_streaks = calculate_streaks(
                player_data,
                stat,
                f'{stat}_CUM_AVG'
            )
            streaks[player_mask] = player_streaks

        df[f'{stat}_ABOVE_AVG_STREAK'] = streaks

    # Fantasy performance trends
    df['DK_TREND_5'] = df.groupby('PLAYER')['DK'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().diff()
    )

    # Game margin categories
    df['BLOWOUT_GAME'] = (abs(df['PLUS MINUS']) > 20).astype(int)

    # Final check for any remaining empty numeric values
    numeric_columns = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    return df


def format_decimals(df, decimal_places=3):
    """Format numeric columns to specified decimal places"""
    for column in df.select_dtypes(include=['float64', 'float32']).columns:
        df[column] = df[column].round(decimal_places)
    return df


def main():
    try:
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

        # Update existing Excel workbook
        print("Updating Excel workbook...")
        try:
            wb = xw.Book(excel_path)
            sheet = wb.sheets['game_logs']
            table = sheet.tables['game_logs']

            # Get the existing table
            data_range = table.data_body_range

            # Calculate the range up to column CA
            last_data_column = 'CA'  # This is the last column before formulas
            if data_range is not None:
                start_row = data_range.row
                end_row = start_row + len(enhanced_df) - 1
                limited_range = sheet.range(f"A{start_row}:{last_data_column}{end_row}")

                # Clear and update only up to column CB
                limited_range.clear_contents()
                limited_range.value = enhanced_df.values

            # Save Excel file
            wb.save()
            wb.close()

            print("Excel table updated successfully!")

        except Exception as e:
            print(f"Error updating Excel file: {str(e)}")
            raise

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