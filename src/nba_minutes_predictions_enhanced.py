import pickle
import pandas as pd
import numpy as np
import xlwings as xw
import os


def apply_high_minutes_curve(minutes, max_minutes):
    """
    Applies a gradual penalty as minutes approach max_minutes.
    The penalty increases more sharply as it gets closer to max.
    """
    if minutes >= 36:  # Only apply curve to high-minute predictions
        # Calculate how close we are to max (0 to 1)
        proximity_to_max = (minutes - 36) / (max_minutes - 36)
        # Apply sigmoid-like curve
        penalty_factor = 1 - (proximity_to_max ** 2 * .5)  # Adjust 0.5 to control curve steepness
        # Apply penalty
        adjusted_minutes = 36 + (minutes - 36) * penalty_factor
        return adjusted_minutes
    return minutes


def ensure_minimum_rotation(team_predictions, team_data, max_mins, min_players=8, min_minutes=8):
    """
    Ensure at least min_players get min_minutes, with priority order:
    1. Players with projection > 0, max minutes > min_minutes, and Last 10 > 1
    2. If needed, add players with only max minutes > min_minutes and Last 10 > 1
    """
    # First try with projected players only
    primary_mask = ((team_data['Last 10 Minutes'] > 1) &
                    (team_data['Max Minutes'] >= min_minutes) &
                    (team_data['Max Minutes'] > 0) &
                    (team_data['Projection'] > 0))
    primary_eligible = team_predictions[primary_mask].index

    # If we don't have enough primary eligible players, consider backup players
    if len(primary_eligible) < min_players:
        backup_mask = ((team_data['Last 10 Minutes'] > 1) &
                       (team_data['Max Minutes'] >= min_minutes) &
                       (team_data['Max Minutes'] > 0) &
                       (~primary_mask))  # Players not in primary group
        backup_eligible = team_predictions[backup_mask].index

        # Combine eligible players, prioritizing primary ones
        eligible_players = list(primary_eligible) + list(backup_eligible)
        eligible_players = eligible_players[:min_players]  # Only take what we need
    else:
        # If we have enough primary eligible players, only use those
        eligible_players = list(primary_eligible)[:min_players]

    # Sort selected players by Last 10 Minutes and apply minimum minutes
    if eligible_players:
        player_priorities = team_data.loc[eligible_players, 'Last 10 Minutes'].sort_values(ascending=False)
        for idx in player_priorities.index:
            if team_predictions[idx] < min_minutes:
                max_allowed = max_mins[idx]
                team_predictions[idx] = min(min_minutes, max_allowed)

    return team_predictions



def adjust_team_minutes_with_minimum_and_boost(predictions_df, min_threshold=8, team_total=240, default_max=38,
                                               boost_threshold=36,  min_rotation=8):
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    # Set predictions under threshold to zero
    adjusted_predictions[adjusted_predictions <= min_threshold] = 0

    # Handle team adjustments
    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_data = predictions_df[team_mask]
        team_predictions = adjusted_predictions[team_mask]
        max_mins = team_data['Max Minutes'].fillna(default_max)

        # First enforce max minutes constraints AND ensure minimum rotation size
        team_predictions = ensure_minimum_rotation(team_predictions, team_data, max_mins, min_rotation, min_threshold)

        # Reapply max minutes constraints
        for idx in team_predictions.index:
            if pd.notnull(max_mins[idx]):
                team_predictions[idx] = min(team_predictions[idx], max_mins[idx])

        if team_predictions.sum() > 0:  # Skip if team has no minutes
            # Get top 5 minute players (only consider those not at max)
            available_for_top = team_predictions[team_predictions < max_mins]
            top_5_idx = available_for_top.nlargest(5).index
            top_5_mins = team_predictions[top_5_idx]

            # Calculate boost for eligible players (under 36 minutes)
            eligible_mask = (top_5_mins < boost_threshold) & (top_5_mins > 0)
            if eligible_mask.any():
                available_boost = sum(boost_threshold - mins for mins in top_5_mins[eligible_mask])

                # Calculate proportional boost
                boost_proportions = (boost_threshold - top_5_mins[eligible_mask]) / available_boost

                # Calculate minutes to redistribute from non-top 5 players
                other_players_idx = team_predictions.index.difference(top_5_idx)
                if len(other_players_idx) > 0:
                    minutes_to_redistribute = min(available_boost * 0.5,
                                                  team_predictions[other_players_idx].sum() * 0.01)

                    # Apply boost to eligible top 5 players
                    for idx in top_5_mins[eligible_mask].index:
                        boost = minutes_to_redistribute * boost_proportions[idx]
                        max_allowed = max_mins[idx]
                        team_predictions[idx] = min(team_predictions[idx] + boost, max_allowed)

                    # Proportionally reduce other players' minutes
                    if minutes_to_redistribute > 0:
                        reduction_factor = 1 - (minutes_to_redistribute / team_predictions[other_players_idx].sum())
                        team_predictions[other_players_idx] *= reduction_factor

            # Scale to 240 while respecting max minutes
            current_total = team_predictions.sum()
            if current_total > 0:  # Avoid division by zero
                remaining_minutes = team_total - current_total
                while abs(remaining_minutes) > 0.1:
                    # Find players who can be adjusted
                    adjustable_mask = (team_predictions > 0) & (team_predictions < max_mins)

                    if not adjustable_mask.any():
                        break

                    adjustable_total = team_predictions[adjustable_mask].sum()
                    if adjustable_total > 0:
                        scale_factor = min(1.2, (adjustable_total + remaining_minutes) / adjustable_total)
                        old_predictions = team_predictions.copy()

                        # Apply scaling
                        team_predictions[adjustable_mask] *= scale_factor

                        # Recheck max minutes
                        for idx in team_predictions.index:
                            if pd.notnull(max_mins[idx]):
                                team_predictions[idx] = min(team_predictions[idx], max_mins[idx])

                        current_total = team_predictions.sum()
                        remaining_minutes = team_total - current_total

                        # Check if we're making progress
                        if (team_predictions == old_predictions).all():
                            break

            # Now apply the curve to high-minute players (but still respect max minutes)
            for idx in team_predictions.index:
                if team_predictions[idx] >= 36:
                    curved_value = apply_high_minutes_curve(team_predictions[idx], max_mins[idx])
                    team_predictions[idx] = min(curved_value, max_mins[idx])

            # Round to 1 decimal place
            team_predictions = np.round(team_predictions, 1)

            # Final adjustment if needed (respecting max minutes)
            if abs(team_predictions.sum() - team_total) > 0.1:
                diff = team_total - team_predictions.sum()
                adjustable_players = team_predictions[(team_predictions > 0) & (team_predictions < max_mins)]
                if len(adjustable_players) > 0:
                    # Distribute difference among adjustable players
                    adjustment_per_player = diff / len(adjustable_players)
                    for idx in adjustable_players.index:
                        new_mins = team_predictions[idx] + adjustment_per_player
                        team_predictions[idx] = round(min(new_mins, max_mins[idx]), 1)

            adjusted_predictions[team_mask] = team_predictions

    return adjusted_predictions


import pickle
import pandas as pd
import numpy as np
import xlwings as xw
import os


# [Keep your existing adjustment functions]
# apply_high_minutes_curve
# ensure_minimum_rotation
# adjust_team_minutes_with_minimum_and_boost

def create_advanced_features(df):
    """Create advanced features for single day predictions"""
    df = df.copy()

    # Create basic required features if they don't exist
    df['MIN'] = df['Minutes']
    #df['PTS'] = df['Projection'] * 0.8  # Rough estimate
    #df['REB'] = df['Projection'] * 0.15  # Rough estimate
    #df['AST'] = df['Projection'] * 0.12  # Rough estimate
    df['DK'] = df['Projection']

    # Create advanced features using available data
    df['MIN_LAST_10_AVG'] = df['Last 10 Minutes']
    df['PTS_LAST_10_AVG'] = df['Last 10 Points']
    df['REB_LAST_10_AVG'] = df['Last 10 Reb']
    df['AST_LAST_10_AVG'] = df['Last 10 Ast']
    df['DK_LAST_10_AVG'] = df['Last 10 DK']

    # Cumulative averages (use Last 10 as proxy)
    df['MIN_CUM_AVG'] = df['MIN']
    df['PTS_CUM_AVG'] = df['PTS']

    # Other features
    df['MIN_TREND'] = 0  # Default to neutral trend
    df['DAYS_REST'] = 1  # Default to 1 day rest
    df['IS_B2B'] = 0  # Default to not back-to-back
    df['IS_HOME'] = 1  # Will be set based on game info

    # Efficiency metrics
    df['PTS_PER_MIN'] = df['PTS'] / df['Minutes'].clip(1)
    df['AST_PER_MIN'] = df['AST'] / df['Minutes'].clip(1)
    df['REB_PER_MIN'] = df['REB'] / df['Minutes'].clip(1)

    # Team averages
    df['MIN_VS_TEAM_AVG'] = 1  # Default to average
    df['MIN_CONSISTENCY'] = 0.1  # Default to stable
    df['MIN_ABOVE_AVG_STREAK'] = 0  # Default to no streak
    df['DK_TREND_5'] = 0  # Default to neutral trend
    df['BLOWOUT_GAME'] = 0  # Default to no blowout

    # New advanced features
    df['MIN_LAST_3_AVG'] = df['Last 10 Minutes']
    df['MIN_LAST_5_AVG'] = df['Last 10 Minutes']
    df['MIN_LAST_7_AVG'] = df['Last 10 Minutes']
    df['MIN_LAST_3_STD'] = 2  # Default variation
    df['MIN_LAST_5_STD'] = 2
    df['MIN_LAST_7_STD'] = 2
    df['MIN_TREND_3'] = 0
    df['MIN_TREND_5'] = 0
    df['MIN_TREND_7'] = 0
    df['ROLE_CHANGE_3_10'] = 0
    df['ROLE_CHANGE_5_10'] = 0
    df['MIN_CONSISTENCY_SCORE'] = 0.1
    df['RECENT_SCORING_EFF'] = df['PTS_PER_MIN']
    df['RECENT_IMPACT'] = 0
    df['FREQ_ABOVE_20'] = (df['Last 10 Minutes'] > 20).astype(float)
    df['FREQ_ABOVE_25'] = (df['Last 10 Minutes'] > 25).astype(float)
    df['FREQ_ABOVE_30'] = (df['Last 10 Minutes'] > 30).astype(float)

    return df


def predict_minutes():
    # Load the trained model
    with open('final_minutes_expanded_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    app = xw.App(visible=False)
    try:
        # Connect to Excel
        excel_path = os.path.join('..', 'dk_import', 'nba_min.xlsm')
        wb = xw.Book(excel_path)
        sheet = wb.sheets['sog_minutes']

        # Get the last row with actual data
        last_row = sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row

        # Read data
        needed_data = {
            'Player': sheet.range(f'A2:A{last_row}').value,
            'Team': sheet.range(f'B2:B{last_row}').value,
            'Salary': sheet.range(f'D2:D{last_row}').value,
            'Minutes': sheet.range(f'E2:E{last_row}').value,
            'Max Minutes': sheet.range(f'J2:J{last_row}').value,
            'Projection': sheet.range(f'M2:M{last_row}').value,
            'Last 10 Minutes': sheet.range(f'N2:N{last_row}').value,
            'PTS': sheet.range(f'R2:R{last_row}').value,
            'REB': sheet.range(f'S2:S{last_row}').value,
            'AST': sheet.range(f'T2:T{last_row}').value,
            'Last 10 Points': sheet.range(f'U2:U{last_row}').value,
            'Last 10 Reb': sheet.range(f'V2:V{last_row}').value,
            'Last 10 Ast': sheet.range(f'W2:W{last_row}').value,
            'Last 10 DK': sheet.range(f'X2:X{last_row}').value,

        }

        # Convert to DataFrame
        data = pd.DataFrame(needed_data)

        # Basic data cleaning
        data = data.dropna(subset=['Team'])
        data = data[data['Team'] != '']

        # Convert numeric columns
        numeric_columns = ['Salary', 'Minutes', 'Projection', 'Max Minutes', 'Last 10 Minutes', 'PTS', 'REB', 'AST',
                           'Last 10 Points',
                           'Last 10 Reb',
                           'Last 10 Ast',
                           'Last 10 DK',

                           ]
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Create DK feature from Projection
        data['DK'] = data['Projection']
        data['DK Name'] = data['Player']

        # Set Minutes to 0 where Projection is 0
        data.loc[data['Projection'] == 0, 'Minutes'] = 0

        # Create advanced features
        enhanced_data = create_advanced_features(data)

        # Define features for prediction (match training features)
        features = [
            'MIN_LAST_10_AVG',
            'PTS_LAST_10_AVG',
            'REB_LAST_10_AVG',
            'AST_LAST_10_AVG',
            'DK_LAST_10_AVG',
            'MIN_CUM_AVG',
            'PTS_CUM_AVG',
            'MIN_TREND',
            'DAYS_REST',
            'IS_B2B', 'IS_HOME', 'PTS_PER_MIN', 'AST_PER_MIN', 'REB_PER_MIN',
            'MIN_VS_TEAM_AVG', 'MIN_CONSISTENCY', 'MIN_ABOVE_AVG_STREAK', 'DK_TREND_5',
            'BLOWOUT_GAME', 'MIN_LAST_3_AVG', 'MIN_LAST_5_AVG', 'MIN_LAST_7_AVG',
            'MIN_LAST_3_STD', 'MIN_LAST_5_STD', 'MIN_LAST_7_STD', 'MIN_TREND_3',
            'MIN_TREND_5', 'MIN_TREND_7', 'ROLE_CHANGE_3_10', 'ROLE_CHANGE_5_10',
            'MIN_CONSISTENCY_SCORE', 'RECENT_SCORING_EFF', 'RECENT_IMPACT',
            'FREQ_ABOVE_20', 'FREQ_ABOVE_25', 'FREQ_ABOVE_30'
        ]

        # Make predictions
        X = enhanced_data[features]
        predictions = model.predict(X)

        # Force minutes to 0 for players with 0 projection
        data['Predicted_Minutes'] = predictions
        data.loc[data['Projection'] == 0, 'Predicted_Minutes'] = 0

        # Apply all adjustments including max minutes constraint
        data['Predicted_Minutes'] = adjust_team_minutes_with_minimum_and_boost(data)

        # Create predictions mapping and write back to Excel
        predictions_dict = dict(zip(data['Player'], data['Predicted_Minutes']))
        predictions_to_write = [predictions_dict.get(player, '') if player else ''
                                for player in needed_data['Player']]
        sheet.range(f'H2').options(transpose=True).value = predictions_to_write

        # Print summary
        print("\nFinal Predictions:")
        for team in sorted(data['Team'].unique()):
            team_data = data[data['Team'] == team]
            print(f"\n{team}:")
            print(f"Team Total: {team_data['Predicted_Minutes'].sum():.1f}")
            print("\nTop Players:")
            top_players = team_data.nlargest(8, 'Predicted_Minutes')
            for _, row in top_players.iterrows():
                print(f"{row['DK Name']:<20} {row['Predicted_Minutes']:.1f}")
                if row['Predicted_Minutes'] >= 36:
                    print(f"  ** High minutes player (Max: {row['Max Minutes']})")

        # Save and close
        wb.save()
        wb.close()

    finally:
        app.quit()


if __name__ == "__main__":
    predict_minutes()