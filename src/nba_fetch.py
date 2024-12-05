import os
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

# Fetch advanced team stats for the regular season
def fetch_advanced_team_stats(season='2023-24', season_type='Regular Season'):
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense='Advanced'
    )
    return team_stats.get_data_frames()[0]


if __name__ == "__main__":
    # Specify the season and season type
    season = '2023-24'
    season_type = 'Regular Season'

    # Fetch the data
    df = fetch_advanced_team_stats(season, season_type)

    # Sort the DataFrame by the 'TEAM_NAME' column
    df_sorted = df.sort_values(by='TEAM_NAME')

    # Determine the script's directory
    script_dir = Path(__file__).resolve().parent  # Get the script's directory

    # Define the relative folder (e.g., child folder "output")
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)  # Create the folder if it doesn't exist

    # Define the output file path
    output_file = output_dir / f'NBA_Advanced_Team_Stats_{season}_Sorted.csv'

    # Save the sorted DataFrame to the relative path
    df_sorted.to_csv(output_file, index=False)

    print(f"File saved to: {output_file}")
