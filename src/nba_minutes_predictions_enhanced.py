import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import xlwings as xw
import os




def get_position_compatibility():
    position_map = {
        'PG': ['PG', 'SG'],
        'SG': ['SG', 'SF', 'PG'],
        'SF': ['SF', 'SG', 'PF'],
        'PF': ['PF', 'SF', 'C'],
        'C': ['C', 'PF'],
        'PG/SG': ['PG', 'SG', 'SF'],
        'SG/SF': ['SG', 'SF', 'PG'],
        'PG/SF': ['PG', 'SF', 'SG'],
        'SF/PF': ['SF', 'PF', 'SG'],
        'PF/C': ['PF', 'C', 'SF']
    }
    return position_map


def apply_position_constraints(predictions_df):
    """Apply position constraints while preserving top players at each position"""
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()
    position_compat = get_position_compatibility()

    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_data = predictions_df[team_mask]
        team_predictions = adjusted_predictions[team_mask]

        total_team_minutes = team_predictions.sum()
        if total_team_minutes == 0:
            continue

        # Target minutes per position should be 20% of total team minutes ±5
        target_pos_minutes = total_team_minutes * 0.2
        min_pos_minutes = max(target_pos_minutes - 5, 0)
        max_pos_minutes = target_pos_minutes + 5

        # Create a list of all players and their eligible positions based on compatibility table
        player_eligibility = {}
        for idx, player in team_data.iterrows():
            listed_pos = player['Position']
            eligible_positions = set()
            # Add all compatible positions based on the mapping
            if listed_pos in position_compat:
                eligible_positions.update(position_compat[listed_pos])
            player_eligibility[idx] = eligible_positions


        # Try to find 5 unique players for the main positions
        protected_players = set()
        position_assignments = {}
        main_positions = ['PG', 'SG', 'SF', 'PF', 'C']

        # First pass: Try to assign highest-minute players to their primary positions
        for pos in main_positions:
            # First try players whose primary position matches
            primary_position_players = [(idx, team_predictions[idx])
                                        for idx in player_eligibility
                                        if team_data.loc[idx, 'Position'].split('/')[0] == pos
                                        and idx not in protected_players
                                        and team_data.loc[idx, 'injury'] not in ['Out', 'Out for season', 'Injured']]


            if primary_position_players :
                # Sort primary position players by minutes
                primary_position_players.sort(key=lambda x: x[1], reverse=True)
                player_idx = primary_position_players[0][0]
                protected_players.add(player_idx)
                position_assignments[pos] = player_idx
            else:
                # If no primary position players, then consider all eligible players
                eligible_players = [(idx, team_predictions[idx])
                                    for idx in player_eligibility
                                    if pos in player_eligibility[idx]
                                    and idx not in protected_players]

                if eligible_players:
                    eligible_players.sort(key=lambda x: x[1], reverse=True)
                    player_idx = eligible_players[0][0]
                    protected_players.add(player_idx)
                    position_assignments[pos] = player_idx

        # If we couldn't find 5 unique players, try to fill remaining positions
        remaining_positions = set(main_positions) - set(position_assignments.keys())
        if remaining_positions:
            for pos in remaining_positions:
                # Look for any player who can play this position and isn't protected
                eligible_players = [(idx, team_predictions[idx])
                                    for idx in player_eligibility
                                    if pos in player_eligibility[idx]]

                if eligible_players:
                    eligible_players.sort(key=lambda x: x[1], reverse=True)
                    player_idx = eligible_players[0][0]
                    protected_players.add(player_idx)
                    position_assignments[pos] = player_idx

        # Debug output
        print(f"\n{team} Protected Players:")
        for pos in main_positions:
            if pos in position_assignments:
                player_idx = position_assignments[pos]
                player = team_data.loc[player_idx]
                print(f"{pos}: {player['Player']} ({player['Position']}) - {team_predictions[player_idx]:.1f} minutes")
                print(player['injury'])
            else:
                print(f"{pos}: No assignment found")

        # Calculate position totals including protected players
        position_totals = {pos: 0 for pos in main_positions}

        # First add protected players' minutes to their assigned positions
        for pos, idx in position_assignments.items():
            position_totals[pos] += team_predictions[idx]

        # Add remaining players' minutes to their eligible positions
        for idx, player in team_data.iterrows():
            if idx not in protected_players:
                eligible_pos = player_eligibility[idx]
                minutes = team_predictions[idx]
                # Distribute minutes equally among eligible positions
                minutes_per_pos = minutes / len(eligible_pos)
                for pos in eligible_pos:
                    if pos in position_totals:
                        position_totals[pos] += minutes_per_pos

        # Only adjust non-protected players when position totals are outside target range
        for pos in position_totals:
            if position_totals[pos] < min_pos_minutes or position_totals[pos] > max_pos_minutes:
                eligible_players = [idx for idx in player_eligibility
                                    if pos in player_eligibility[idx] and
                                    idx not in protected_players]

                if eligible_players:
                    if position_totals[pos] < min_pos_minutes:
                        deficit = min_pos_minutes - position_totals[pos]
                        increase_per_player = deficit / len(eligible_players)
                        for idx in eligible_players:
                            team_predictions[idx] += increase_per_player
                    elif position_totals[pos] > max_pos_minutes:
                        excess = position_totals[pos] - max_pos_minutes
                        decrease_per_player = excess / len(eligible_players)
                        for idx in eligible_players:
                            team_predictions[idx] = max(0, team_predictions[idx] - decrease_per_player)

        adjusted_predictions[team_mask] = team_predictions

    return adjusted_predictions



def smooth_adjustments(original_predictions, adjusted_predictions, smoothing_factor=0.5):
    return original_predictions * (1 - smoothing_factor) + adjusted_predictions * smoothing_factor


def adjust_predictions(predictions, players_df):
    position_map = get_position_compatibility()
    adjusted_predictions = predictions.copy()

    # Create position groups
    position_totals = {pos: 0 for pos in ['PG', 'SG', 'SF', 'PF', 'C']}

    # First pass: assign minutes to primary positions
    for idx, player in players_df.iterrows():
        primary_pos = player['Position'].split('/')[0]
        position_totals[primary_pos] += predictions[idx]

    # Adjust if position totals are outside bounds (45-50 minutes)
    for pos in position_totals:
        if position_totals[pos] < 45:
            # Find players who can play this position and increase their minutes
            for idx, player in players_df.iterrows():
                if pos in position_map[player['Position']]:
                    deficit = 45 - position_totals[pos]
                    increase = min(deficit, 48 - adjusted_predictions[idx])
                    adjusted_predictions[idx] += increase
                    position_totals[pos] += increase

        elif position_totals[pos] > 50:
            # Reduce minutes proportionally for players in this position
            excess = position_totals[pos] - 50
            players_in_pos = players_df[players_df['Position'].str.contains(pos)].index
            if len(players_in_pos) > 0:
                reduction_per_player = excess / len(players_in_pos)
                for idx in players_in_pos:
                    adjusted_predictions[idx] = max(0, adjusted_predictions[idx] - reduction_per_player)

    # Ensure total minutes = 240
    total_minutes = sum(adjusted_predictions)
    if total_minutes != 240:
        scale_factor = 240 / total_minutes
        adjusted_predictions *= scale_factor

    return adjusted_predictions

def apply_high_minutes_curve(current_total, minutes, max_minutes, min_minutes = 32):
    """
    Applies a gradual penalty as minutes approach max_minutes.
    The penalty increases more sharply as it gets closer to max.
    """
    min_minutes = min_minutes / 240 * current_total
    max_minutes = max_minutes / 240 * current_total

    if minutes >= min_minutes and max_minutes > min_minutes:  # Only apply curve to high-minute predictions
        # Calculate how close we are to max (0 to 1)
        proximity_to_max = (minutes - min_minutes) / (max_minutes - min_minutes)
        # Apply sigmoid-like curve
        penalty_factor = 1 - (proximity_to_max ** 2 * .5)  # Adjust 0.5 to control curve steepness
        # Apply penalty
        adjusted_minutes = min_minutes + (minutes - min_minutes) * penalty_factor
        return adjusted_minutes
    return minutes


def ensure_minimum_rotation(team_predictions, team_data, max_mins, min_players=8, min_minutes=8):
    """
    Ensure at least min_players get min_minutes, with priority order:
    1. Players with projection > 0, max minutes > min_minutes, and Last 10 > 1
    2. If needed, add players with only max minutes > min_minutes and Last 10 > 1
    """
    # First try with projected players only
    primary_mask = ((team_data['MIN_LAST_10_AVG'] > 1) &
                    (team_data['Max Minutes'] >= min_minutes) &
                    (team_data['Max Minutes'] > 0) &
                    (team_data['Projection'] >= 0))#zero
    primary_eligible = team_predictions[primary_mask].index

    # If we don't have enough primary eligible players, consider backup players
    if len(primary_eligible) < min_players:
        backup_mask = ((team_data['MIN_LAST_10_AVG'] > 1) &
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
        player_priorities = team_data.loc[eligible_players, 'MIN_LAST_10_AVG'].sort_values(ascending=False)
        for idx in player_priorities.index:
            if team_predictions[idx] < min_minutes:
                max_allowed = max_mins[idx]
                team_predictions[idx] = min(min_minutes, max_allowed)

    return team_predictions


def adjust_team_minutes_with_minimum_and_boost(predictions_df, min_threshold=8, team_total=240, default_max=38,
                                               boost_threshold=36, min_rotation=8):
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    # Create a mask for players that should be forced to zero
    #force_zero_mask = (predictions_df['Projection'] == 0) | (predictions_df['Minutes'] <= 6)
    force_zero_mask = (predictions_df['Projection'] < 0)#zero

    # Set initial zeros based on force_zero_mask
    adjusted_predictions[force_zero_mask] = 0

    # Set predictions under threshold to zero, but respect force_zero_mask
    adjusted_predictions[(adjusted_predictions <= min_threshold) & ~force_zero_mask] = 0

    # Handle team adjustments
    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_data = predictions_df[team_mask]
        team_predictions = adjusted_predictions[team_mask]
        max_mins = team_data['Max Minutes'].fillna(default_max)

        # Respect forced zeros
        team_force_zero = force_zero_mask[team_mask]
        team_predictions[team_force_zero] = 0

        # First enforce max minutes constraints AND ensure minimum rotation size
        min_threshold = 8
        team_predictions = ensure_minimum_rotation(team_predictions, team_data, max_mins, min_rotation, min_threshold)

        # Reapply max minutes constraints
        for idx in team_predictions.index:
            if pd.notnull(max_mins[idx]):
                team_predictions[idx] = min(team_predictions[idx], max_mins[idx])

        if team_predictions.sum() > 0:  # Skip if team has no minutes
            # Get top 5 minute players (only consider those not at max and not forced zero)
            available_for_top = team_predictions[
                (team_predictions < max_mins) &
                ~team_force_zero
                ]
            top_5_idx = available_for_top.nlargest(5).index
            top_5_mins = team_predictions[top_5_idx]

            # Calculate boost for eligible players (under 36 minutes)
            boost_threshold = 0
            eligible_mask = (top_5_mins < boost_threshold) & (top_5_mins > 0)
            if eligible_mask.any():
                available_boost = sum(boost_threshold - mins for mins in top_5_mins[eligible_mask])

                # Calculate proportional boost
                boost_proportions = (boost_threshold - top_5_mins[eligible_mask]) / available_boost

                # Calculate minutes to redistribute from non-top 5 players
                other_players_idx = team_predictions.index.difference(top_5_idx)
                if len(other_players_idx) > 0:
                    minutes_to_redistribute = min(available_boost * 0.5,
                                                  team_predictions[other_players_idx].sum() * 0.1)

                    # Apply boost to eligible top 5 players
                    for idx in top_5_mins[eligible_mask].index:
                        boost = minutes_to_redistribute * boost_proportions[idx]
                        max_allowed = max_mins[idx]
                        team_predictions[idx] = min(team_predictions[idx] + boost, max_allowed)

                    # Proportionally reduce other players' minutes
                    if minutes_to_redistribute > 0:
                        reduction_factor = 1 - (minutes_to_redistribute / team_predictions[other_players_idx].sum())
                        team_predictions[other_players_idx] *= reduction_factor

            # Modified scaling loop
            current_total = team_predictions.sum()
            iteration_count = 0
            max_iterations = 50

            while abs(current_total - team_total) > 0.1 and iteration_count < max_iterations:
                adjustable_mask = (team_predictions > 0) & (team_predictions < max_mins) & ~team_force_zero

                if not adjustable_mask.any():
                    break

                old_predictions = team_predictions.copy()

                # Get adjustable values
                adjustable_values = team_predictions[adjustable_mask].values
                #
                # # Apply non-linear scaling to adjustable values
                # new_values = nonlinear_scale(
                #     adjustable_values,
                #     team_total - team_predictions[~adjustable_mask].sum(),  # target for adjustable players
                #     adjustable_values.sum(),  # current total for adjustable players
                #     method='log',  # Try 'log', 'sqrt', or 'exp'
                #     intensity=1.0  # Adjust this parameter to control scaling intensity
                # )

                # Update predictions
                team_predictions[adjustable_mask] = adjustable_values

                # Recheck max minutes
                for idx in team_predictions.index:
                    if pd.notnull(max_mins[idx]):
                        team_predictions[idx] = min(team_predictions[idx], max_mins[idx])

                # Ensure zeros stay zero
                team_predictions[team_force_zero] = 0

                current_total = team_predictions.sum()
                iteration_count += 1

                if (abs(team_predictions - old_predictions) < 0.01).all():
                    break

            # Now apply the curve to high-minute players
            min_minutes = 32
            for idx in team_predictions.index:
                if team_predictions[idx] >= min_minutes / 240 * current_total:
                    curved_value = apply_high_minutes_curve(current_total, team_predictions[idx], max_mins[idx], min_minutes)
                    team_predictions[idx] = min(curved_value, max_mins[idx])

            # Round to 1 decimal place
            team_predictions = np.round(team_predictions, 1)

            # Final adjustment if needed (respecting max minutes)
            if abs(team_predictions.sum() - team_total) > 0.1:
                diff = team_total - team_predictions.sum()
                adjustable_players = team_predictions[
                    (team_predictions > 0) &
                    (team_predictions < max_mins) &
                    ~team_force_zero
                    ]
                if len(adjustable_players) > 0:
                    adjustment_per_player = diff / len(adjustable_players)
                    for idx in adjustable_players.index:
                        new_mins = team_predictions[idx] + adjustment_per_player
                        team_predictions[idx] = round(min(new_mins, max_mins[idx]), 1)


        adjusted_predictions[team_mask] = team_predictions

    # Final check to ensure forced zeros remain zero
    adjusted_predictions[force_zero_mask] = 0

    return adjusted_predictions





def create_advanced_features(df):
    """Create advanced features for single day predictions"""




    df = df.copy()
    #df['MIN_VS_TEAM_AVG'] = df['DARKO Minutes']
    # Create basic required features if they don't exist
    #df['MIN'] = df['Minutes']
    #df['DK'] = df['Projection']

    # Create advanced features using available data
    #df['MIN_LAST_10_AVG'] = df['Last 10 Minutes']
    #df['PTS_LAST_10_AVG'] = df['Last 10 Points']
    #df['REB_LAST_10_AVG'] = df['Last 10 Reb']
    #df['AST_LAST_10_AVG'] = df['Last 10 Ast']
    #df['DK_LAST_10_AVG'] = df['Last 10 DK']


    # Other features
    df['DAYS_REST'] = 1  # Default to 1 day rest
    df['IS_B2B'] = 0  # Default to not back-to-back
    #df['IS_HOME'] = 1  # Will be set based on game info

    # Efficiency metrics
    #df['PTS_PER_MIN'] = df['PTS'] / df['Minutes'].clip(1)
    #df['AST_PER_MIN'] = df['AST'] / df['Minutes'].clip(1)
    #df['REB_PER_MIN'] = df['REB'] / df['Minutes'].clip(1)

    df['BLOWOUT_GAME'] = 0  # Default to no blowout

    # New advanced features
    #df['MIN_LAST_3_AVG'] = df['Last 10 Minutes']
    #df['MIN_LAST_5_AVG'] = df['Last 10 Minutes']
    #df['MIN_LAST_7_AVG'] = df['Last 10 Minutes']
    df['MIN_LAST_3_STD'] = 2  # Default variation
    df['MIN_LAST_5_STD'] = 2
    df['MIN_LAST_7_STD'] = 2
    #df['MIN_TREND_3'] = 0
    #df['MIN_TREND_5'] = 0
    #df['MIN_TREND_7'] = 0
    #df['ROLE_CHANGE_3_10'] = 0
    #df['ROLE_CHANGE_5_10'] = 0
    #df['MIN_CONSISTENCY_SCORE'] = 0.1
    #df['RECENT_SCORING_EFF'] = df['PTS_PER_MIN']
    df['RECENT_IMPACT'] = 0
    #df['FREQ_ABOVE_20'] = (df['Last 10 Minutes'] > 20).astype(float)
    #df['FREQ_ABOVE_25'] = (df['Last 10 Minutes'] > 25).astype(float)
    #df['FREQ_ABOVE_30'] = (df['Last 10 Minutes'] > 30).astype(float)

    return df



class PredictMinutes:
    def __int__(self):
        pass


    def predict_minutes_df(self, df):
        # Increase display options for columns

        print("****************************************")
        current_folder = Path(__file__).parent  # Current script directory (src)
        target_folder = current_folder.parent / "src"  # Sibling folder (dk-import)
        os.chdir(target_folder)
        print(f"Current working directory: {os.getcwd()}")
        with open('final_minutes_expanded_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
            data = df
            # Basic data cleaning
            data = data.dropna(subset=['Team'])
            data = data[data['Team'] != '']
            # Set Minutes to 0 where Projection is 0
            #data.loc[data['Projection'] == 0, 'Minutes'] = 0
            # Create advanced features
            enhanced_data = create_advanced_features(data)

            # Define features for prediction (match training features)
            features = [
                #'MIN_VS_TEAM_AVG',
                'MIN_CUM_AVG',
                'MIN_ABOVE_AVG_STREAK',
                'MIN_LAST_10_AVG',
                'MIN_CONSISTENCY',

                #  'DK_TREND_5',
                #  'DK_LAST_10_AVG',
                # 'PTS_CUM_AVG',
                # 'REB_PER_MIN',
                # 'AST_PER_MIN',
                # 'PTS_PER_MIN',
                # 'AST_LAST_10_AVG',
                # 'REB_LAST_10_AVG',
                #
                # 'DAYS_REST',
                # 'PTS_LAST_10_AVG',
                #'BLOWOUT_GAME',
                #'IS_HOME',
                #'MIN_TREND',
                #'IS_B2B',

                # New advanced features
                'MIN_LAST_3_AVG',
                'MIN_LAST_5_AVG',
                # 'ROLE_CHANGE_3_10',
                # 'ROLE_CHANGE_5_10',
                # 'MIN_CONSISTENCY_SCORE',
                # 'RECENT_SCORING_EFF',
                # 'RECENT_IMPACT',
                #
                # 'FREQ_ABOVE_20',
                # 'FREQ_ABOVE_25',
                # 'FREQ_ABOVE_30',
                #
                #  'TEAM_PROJ_RANK',
                #  #'IS_TOP_3_PROJ',
                 'TEAM_MIN_PERCENTAGE',
                #  'LOW_MIN_TOP_PLAYER',
                 'Projection'

            ]


            enhanced_data['TEAM_MIN_PERCENTAGE'] = enhanced_data.groupby('Team')['Minutes'].transform(lambda x: x / x.sum() * 100)

            # Make predictions
            X = enhanced_data[features]
            raw_predictions = model.predict(X)

            # Store original predictions
            data['Original_Minutes'] = raw_predictions

            # Calculate team total minutes for each team
            team_total_minutes = data.groupby('Team')['Minutes'].sum()

            # Apply minimum minutes requirement based on team's actual total minutes
            data['Min_Required_Minutes'] = data.apply(
                lambda row: (row['Min Minutes'] / 240 * team_total_minutes[row['Team']])
                if pd.notnull(row['Min Minutes']) else 0,
                axis=1
            )

            # Adjust raw predictions to meet minimum requirements
            adjusted_predictions = raw_predictions.copy()
            for idx, row in data.iterrows():
                if pd.notnull(row['Min Minutes']) and row[
                    'Min Minutes'] > 0:  # Check if player has min minutes requirement
                    adjusted_predictions[idx] = max(adjusted_predictions[idx], row['Min_Required_Minutes'])

            data['Predicted_Minutes'] = adjusted_predictions


            # Apply zero conditions
            #zero_mask = (data['Projection'] == 0)
            #data.loc[zero_mask, 'Predicted_Minutes'] = 0

            # Apply position constraints to the already-modified predictions
            print('Stage A:  Predicted ', data[data['Player'] == 'Darius Garland']['Predicted_Minutes'])
            data['Predicted_Minutes'] = apply_position_constraints(data)
            data['Predicted_Minutes'] = adjust_team_minutes_with_minimum_and_boost(data)

            # Print summary
            print("\nFinal Predictions:")
            for team in sorted(data['Team'].unique()):
                team_data = data[data['Team'] == team]
                print(f"\n{team}:")
                print(f"Team Total: {team_data['Predicted_Minutes'].sum():.1f}")
                print("\nTop Players:")
                top_players = team_data.nlargest(10, 'Predicted_Minutes')
                for _, row in top_players.iterrows():
                    print(f"{row['Player']:<20} {row['Predicted_Minutes']:.1f}")
                    if row['Predicted_Minutes'] >= 36:
                        print(f"  ** High minutes player (Max: {row['Max Minutes']})")

        return data


if __name__ == "__main__":
    predictions = PredictMinutes()
    predictions.predict_minutes_df()