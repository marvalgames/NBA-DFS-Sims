import pickle
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
        min_pos_minutes = max(target_pos_minutes - 8, 0)
        max_pos_minutes = target_pos_minutes + 8

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
            eligible_players = [(idx, team_predictions[idx])
                                for idx in player_eligibility
                                if pos in player_eligibility[idx] and idx not in protected_players]

            if eligible_players:
                # Sort by minutes descending
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
    df['MIN_LAST_7_AVG'] = df['Last 10 Minutes']
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


def predict_minutes():
    # Load the trained model
    with open('final_minutes_expanded_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    app = xw.App(visible=False)
    try:
        # Connect to Excel
        excel_path = os.path.join('..', 'dk_import', 'nba_merged.xlsm')
        wb = xw.Book(excel_path)
        sheet = wb.sheets['sog_minutes']

        # Get the last row with actual data
        last_row = sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row
        needed_data = {
            'Player': sheet.range(f'A2:A{last_row}').value,
            'Team': sheet.range(f'B2:B{last_row}').value,
            'Position': sheet.range(f'C2:C{last_row}').value,
            'Salary': sheet.range(f'D2:D{last_row}').value,
            'Minutes': sheet.range(f'E2:E{last_row}').value,
            'BBM Minutes': sheet.range(f'F2:F{last_row}').value,
            'FTA Minutes': sheet.range(f'G2:G{last_row}').value,
            'SOG Minutes': sheet.range(f'H2:H{last_row}').value,
            'Minutes Played': sheet.range(f'I2:I{last_row}').value,
            'Max Minutes': sheet.range(f'J2:J{last_row}').value,
            'FTA Own': sheet.range(f'K2:K{last_row}').value,
            'SOG Own': sheet.range(f'L2:L{last_row}').value,
            'Projection': sheet.range(f'M2:M{last_row}').value,
            'Last 10 Minutes': sheet.range(f'N2:N{last_row}').value,
            'Name': sheet.range(f'O2:O{last_row}').value,
            'MSE': sheet.range(f'P2:P{last_row}').value,
            'DARKO Minutes': sheet.range(f'Q2:Q{last_row}').value,
            'Minutes Override': sheet.range(f'R2:R{last_row}').value,
            'MIN_CUM_AVG': sheet.range(f'S2:S{last_row}').value,
            'MIN_LAST_3_AVG': sheet.range(f'T2:T{last_row}').value,
            'MIN_LAST_5_AVG': sheet.range(f'U2:U{last_row}').value,
            'MIN_LAST_10_AVG': sheet.range(f'V2:V{last_row}').value,
            'MIN_TREND': sheet.range(f'W2:W{last_row}').value,
            'MIN_CONSISTENCY': sheet.range(f'X2:X{last_row}').value,
            'MIN_CONSISTENCY_SCORE': sheet.range(f'Y2:Y{last_row}').value,
            'MIN_ABOVE_20': sheet.range(f'Z2:Z{last_row}').value,
            'MIN_ABOVE_25': sheet.range(f'AA2:AA{last_row}').value,
            'MIN_ABOVE_30': sheet.range(f'AB2:AB{last_row}').value,
            'MIN_ABOVE_AVG_STREAK': sheet.range(f'AC2:AC{last_row}').value,
            'FREQ_ABOVE_20': sheet.range(f'AD2:AD{last_row}').value,
            'FREQ_ABOVE_25': sheet.range(f'AE2:AE{last_row}').value,
            'FREQ_ABOVE_30': sheet.range(f'AF2:AF{last_row}').value,
            'Projected Pts': sheet.range(f'AG2:AG{last_row}').value,
            'DK': sheet.range(f'AH2:AH{last_row}').value,
            'DK_CUM_AVG': sheet.range(f'AI2:AI{last_row}').value,
            'DK_LAST_3_AVG': sheet.range(f'AJ2:AJ{last_row}').value,
            'DK_LAST_5_AVG': sheet.range(f'AK2:AK{last_row}').value,
            'DK_LAST_10_AVG': sheet.range(f'AL2:AL{last_row}').value,
            'DK_TREND': sheet.range(f'AM2:AM{last_row}').value,
            'DK_TREND_5': sheet.range(f'AN2:AN{last_row}').value,
            'DK_CONSISTENCY': sheet.range(f'AO2:AO{last_row}').value,
            'PTS_LAST_3_AVG': sheet.range(f'AP2:AP{last_row}').value,
            'PTS_LAST_5_AVG': sheet.range(f'AQ2:AQ{last_row}').value,
            'PTS_LAST_10_AVG': sheet.range(f'AR2:AR{last_row}').value,
            'REB_LAST_3_AVG': sheet.range(f'AS2:AS{last_row}').value,
            'REB_LAST_5_AVG': sheet.range(f'AT2:AT{last_row}').value,
            'REB_LAST_10_AVG': sheet.range(f'AU2:AU{last_row}').value,
            'AST_LAST_3_AVG': sheet.range(f'AV2:AV{last_row}').value,
            'AST_LAST_5_AVG': sheet.range(f'AW2:AW{last_row}').value,
            'AST_LAST_10_AVG': sheet.range(f'AX2:AX{last_row}').value,
            'PTS_CUM_AVG': sheet.range(f'AY2:AY{last_row}').value,
            'REB_CUM_AVG': sheet.range(f'AZ2:AZ{last_row}').value,
            'AST_CUM_AVG': sheet.range(f'BA2:BA{last_row}').value,
            'PTS_PER_MIN': sheet.range(f'BB2:BB{last_row}').value,
            'REB_PER_MIN': sheet.range(f'BC2:BC{last_row}').value,
            'AST_PER_MIN': sheet.range(f'BD2:BD{last_row}').value,
            'SCORING_EFFICIENCY': sheet.range(f'BE2:BE{last_row}').value,
            'RECENT_SCORING_EFF': sheet.range(f'BF2:BF{last_row}').value,
            'FG_PCT': sheet.range(f'BG2:BG{last_row}').value,
            'FG3_PCT': sheet.range(f'BH2:BH{last_row}').value,
            'FT_PCT': sheet.range(f'BI2:BI{last_row}').value,
            'FGM': sheet.range(f'BJ2:BJ{last_row}').value,
            'FGA': sheet.range(f'BK2:BK{last_row}').value,
            'FG3M': sheet.range(f'BL2:BL{last_row}').value,
            'FG3A': sheet.range(f'BM2:BM{last_row}').value,
            'FTM': sheet.range(f'BN2:BN{last_row}').value,
            'FTA': sheet.range(f'BO2:BO{last_row}').value,
            'PLUS_MINUS': sheet.range(f'BP2:BP{last_row}').value,
            'PLUS_MINUS_PER_MIN': sheet.range(f'BQ2:BQ{last_row}').value,
            'PLUS MINUS_LAST_3_AVG': sheet.range(f'BR2:BR{last_row}').value,
            'PLUS MINUS_LAST_5_AVG': sheet.range(f'BS2:BS{last_row}').value,
            'PLUS MINUS_LAST_10_AVG': sheet.range(f'BT2:BT{last_row}').value,
            'PLUS MINUS_CUM_AVG': sheet.range(f'BU2:BU{last_row}').value,
            'PLUS MINUS_TREND': sheet.range(f'BV2:BV{last_row}').value,
            'PLUS MINUS_CONSISTENCY': sheet.range(f'BW2:BW{last_row}').value,
            'PTS_TREND': sheet.range(f'BX2:BX{last_row}').value,
            'REB_TREND': sheet.range(f'BY2:BY{last_row}').value,
            'AST_TREND': sheet.range(f'BZ2:BZ{last_row}').value,
            'PTS_CONSISTENCY': sheet.range(f'CA2:CA{last_row}').value,
            'REB_CONSISTENCY': sheet.range(f'CB2:CB{last_row}').value,
            'AST_CONSISTENCY': sheet.range(f'CC2:CC{last_row}').value,
            'TEAM_MIN_PERCENTAGE': sheet.range(f'CD2:CD{last_row}').value,
            'TEAM_PROJ_RANK': sheet.range(f'CE2:CE{last_row}').value,
            'PTS_VS_TEAM_AVG': sheet.range(f'CF2:CF{last_row}').value,
            'REB_VS_TEAM_AVG': sheet.range(f'CG2:CG{last_row}').value,
            'AST_VS_TEAM_AVG': sheet.range(f'CH2:CH{last_row}').value,
            'MIN_VS_TEAM_AVG': sheet.range(f'CI2:CI{last_row}').value,
            'ROLE_CHANGE_3_10': sheet.range(f'CJ2:CJ{last_row}').value,
            'ROLE_CHANGE_5_10': sheet.range(f'CK2:CK{last_row}').value,
            'IS_TOP_3_PROJ': sheet.range(f'CL2:CL{last_row}').value,
            'LOW_MIN_TOP_PLAYER': sheet.range(f'CM2:CM{last_row}').value,
            'GAME_DATE': sheet.range(f'CN2:CN{last_row}').value,
            'IS_HOME': sheet.range(f'CO2:CO{last_row}').value,
            'IS_B2B': sheet.range(f'CP2:CP{last_row}').value,
            'DAYS_REST': sheet.range(f'CQ2:CQ{last_row}').value,
            'MATCHUP': sheet.range(f'CR2:CR{last_row}').value,
            'WL': sheet.range(f'CS2:CS{last_row}').value,
            'BLOWOUT_GAME': sheet.range(f'CT2:CT{last_row}').value,
            'STL': sheet.range(f'CU2:CU{last_row}').value,
            'BLK': sheet.range(f'CV2:CV{last_row}').value,
            'TOV': sheet.range(f'CW2:CW{last_row}').value,
            'OREB': sheet.range(f'CX2:CX{last_row}').value,
            'DREB': sheet.range(f'CY2:CY{last_row}').value,
            'PF': sheet.range(f'CZ2:CZ{last_row}').value,
            'PLAYER_ID': sheet.range(f'DA2:DA{last_row}').value,
            'TEAM_ID': sheet.range(f'DB2:DB{last_row}').value,
            'TEAM_NAME': sheet.range(f'DC2:DC{last_row}').value,
            'GAME_ID': sheet.range(f'DD2:DD{last_row}').value,
            'SEASON_ID': sheet.range(f'DE2:DE{last_row}').value,
            'VIDEO_AVAILABLE': sheet.range(f'DF2:DF{last_row}').value,
        }


        # Convert to DataFrame
        data = pd.DataFrame(needed_data)


        # Basic data cleaning
        data = data.dropna(subset=['Team'])
        data = data[data['Team'] != '']

        # Convert numeric columns
        numeric_columns = [
            'Salary',
            'Minutes',
            'BBM Minutes',
            'FTA Minutes',
            'SOG Minutes',
            'Minutes Played',
            'Max Minutes',
            'FTA Own',
            'SOG Own',
            'Projection',
            'Last 10 Minutes',
            'MSE',
            'DARKO Minutes',
            'Minutes Override',
            'MIN_CUM_AVG',
            'MIN_LAST_3_AVG',
            'MIN_LAST_5_AVG',
            'MIN_LAST_10_AVG',
            'MIN_ABOVE_AVG_STREAK',
            'MIN_TREND',
            'MIN_CONSISTENCY',
            'MIN_CONSISTENCY_SCORE',
            'MIN_ABOVE_20',
            'MIN_ABOVE_25',
            'MIN_ABOVE_30',
            'FREQ_ABOVE_20',
            'FREQ_ABOVE_25',
            'FREQ_ABOVE_30',
            'Projected Pts',
            'DK',
            'DK_CUM_AVG',
            'DK_LAST_3_AVG',
            'DK_LAST_5_AVG',
            'DK_LAST_10_AVG',
            'DK_TREND',
            'DK_TREND_5',
            'DK_CONSISTENCY',
            'PTS_LAST_3_AVG',
            'PTS_LAST_5_AVG',
            'PTS_LAST_10_AVG',
            'REB_LAST_3_AVG',
            'REB_LAST_5_AVG',
            'REB_LAST_10_AVG',
            'AST_LAST_3_AVG',
            'AST_LAST_5_AVG',
            'AST_LAST_10_AVG',
            'PTS_CUM_AVG',
            'REB_CUM_AVG',
            'AST_CUM_AVG',
            'PTS_PER_MIN',
            'REB_PER_MIN',
            'AST_PER_MIN',
            'SCORING_EFFICIENCY',
            'RECENT_SCORING_EFF',
            'FG_PCT',
            'FG3_PCT',
            'FT_PCT',
            'FGM',
            'FGA',
            'FG3M',
            'FG3A',
            'FTM',
            'FTA',
            'PLUS_MINUS',
            'PLUS_MINUS_PER_MIN',
            'PLUS MINUS_LAST_3_AVG',
            'PLUS MINUS_LAST_5_AVG',
            'PLUS MINUS_LAST_10_AVG',
            'PLUS MINUS_CUM_AVG',
            'PLUS MINUS_TREND',
            'PLUS MINUS_CONSISTENCY',
            'PTS_TREND',
            'REB_TREND',
            'AST_TREND',
            'PTS_CONSISTENCY',
            'REB_CONSISTENCY',
            'AST_CONSISTENCY',
            'TEAM_MIN_PERCENTAGE',
            'TEAM_PROJ_RANK',
            'PTS_VS_TEAM_AVG',
            'REB_VS_TEAM_AVG',
            'AST_VS_TEAM_AVG',
            'MIN_VS_TEAM_AVG',
            'ROLE_CHANGE_3_10',
            'ROLE_CHANGE_5_10',
            'DAYS_REST',
            'STL',
            'BLK',
            'TOV',
            'OREB',
            'DREB',
            'PF'
        ]

        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        # Create DK feature from Projection
        #data['DK'] = data['Projection']
        #data['DK Name'] = data['Player']

        # Set Minutes to 0 where Projection is 0
        data.loc[data['Projection'] == 0, 'Minutes'] = 0

        # Create advanced features
        enhanced_data = create_advanced_features(data)

        # Define features for prediction (match training features)
        features = [
            'MIN_VS_TEAM_AVG',
            'MIN_CUM_AVG',
            'MIN_ABOVE_AVG_STREAK',
            'MIN_LAST_10_AVG',
            'MIN_CONSISTENCY',

             'DK_TREND_5',
             'DK_LAST_10_AVG',
            # 'PTS_CUM_AVG',
            # 'REB_PER_MIN',
            # 'AST_PER_MIN',
            # 'PTS_PER_MIN',
            # 'AST_LAST_10_AVG',
            # 'REB_LAST_10_AVG',
            #
            # 'DAYS_REST',
            # 'PTS_LAST_10_AVG',
            # 'BLOWOUT_GAME',
            # 'IS_HOME',
            # 'MIN_TREND',
            # 'IS_B2B',
            #
            # # New advanced features
            # 'MIN_LAST_3_AVG',
            # 'MIN_LAST_5_AVG',
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
             'TEAM_PROJ_RANK',
             'IS_TOP_3_PROJ',
             'TEAM_MIN_PERCENTAGE',
             'LOW_MIN_TOP_PLAYER',
             'Projection'




        ]

        # After reading data but before predictions
        print("Before override - AD's MIN_VS_TEAM_AVG:",
              enhanced_data.loc[enhanced_data['Player'] == 'Anthony Davis', 'TEAM_MIN_PERCENTAGE'].values[0])


        # Override the value
        enhanced_data['TEAM_MIN_PERCENTAGE'] = enhanced_data.groupby('Team')['DARKO Minutes'].transform(
            lambda x: x / x.sum() * 100
        )

        enhanced_data['MIN_VS_TEAM_AVG'] = enhanced_data.groupby('Team')['DARKO Minutes'].transform(
            lambda x: x / x.mean()
        )

        print("After override - AD's MIN_VS_TEAM_AVG:",
              enhanced_data.loc[enhanced_data['Player'] == 'Anthony Davis', 'TEAM_MIN_PERCENTAGE'].values[0])


        # Make predictions
        X = enhanced_data[features]

        raw_predictions = model.predict(X)

        # Set initial zeros
        #raw_predictions[data['Projection'] == 0] = 0
        data['Predicted_Minutes'] = raw_predictions


        # In your main prediction code
        data['Original_Minutes'] = raw_predictions
        data['Predicted_Minutes'] = apply_position_constraints(data)

        # Debug output
        #for team in data['Team'].unique():
            #team_mask = data['Team'] == team
            #team_data = data[team_mask]
            #print(f"\n{team}:")

        # Force minutes to 0 for players with 0 projection
        data.loc[data['Projection'] == 0, 'Predicted_Minutes'] = 0

        # Add new conditions for DARKO Minutes and Minutes
        data.loc[data['Minutes'] <= 6, 'Predicted_Minutes'] = 0
        #data.loc[data['DARKO Minutes'] <= 6, 'Predicted_Minutes'] = 0  # Assuming 'Projection' is DARKO Minutes

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
                print(f"{row['Player']:<20} {row['Predicted_Minutes']:.1f}")
                if row['Predicted_Minutes'] >= 36:
                    print(f"  ** High minutes player (Max: {row['Max Minutes']})")

        # Save and close
        wb.save()
        wb.close()

    finally:
        app.quit()


if __name__ == "__main__":
    predict_minutes()