import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os
from tabulate import tabulate




def smooth_adjustments(original_predictions, adjusted_predictions, smoothing_factor=0.5):
    return (original_predictions * (1 - smoothing_factor) + adjusted_predictions * smoothing_factor)


def adjust_team_minutes(predictions_df):
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()
    for team in predictions_df['TEAM'].unique():
        team_mask = predictions_df['TEAM'] == team
        team_total = adjusted_predictions[team_mask].sum()
        if team_total > 0:
            scale_factor = 240 / team_total
            adjusted_predictions[team_mask] = adjusted_predictions[team_mask] * scale_factor
    return adjusted_predictions


def adjust_team_minutes_with_constraints(predictions_df, min_minutes=0, max_minutes=48):
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()
    for team in predictions_df['TEAM'].unique():
        team_mask = predictions_df['TEAM'] == team
        team_predictions = adjusted_predictions[team_mask]
        team_total = team_predictions.sum()
        if team_total > 0:
            scale_factor = 240 / team_total
            scaled_predictions = team_predictions * scale_factor
            scaled_predictions = scaled_predictions.clip(min_minutes, max_minutes)
            final_scale = 240 / scaled_predictions.sum()
            adjusted_predictions[team_mask] = scaled_predictions * final_scale
    return adjusted_predictions


def adjust_team_minutes_with_minimum(predictions_df, min_threshold=8, team_total=240):
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()
    adjusted_predictions[adjusted_predictions <= min_threshold] = 0
    for team in predictions_df['TEAM'].unique():
        team_mask = predictions_df['TEAM'] == team
        team_total_mins = adjusted_predictions[team_mask].sum()
        if team_total_mins > 0:
            non_zero_mask = (predictions_df['TEAM'] == team) & (adjusted_predictions > 0)
            scale_factor = team_total / team_total_mins
            adjusted_predictions[non_zero_mask] *= scale_factor
            team_predictions = adjusted_predictions[non_zero_mask]
            rounded_predictions = np.round(team_predictions, 1)
            rounded_total = rounded_predictions.sum()
            if rounded_total != team_total:
                diff = team_total - rounded_total
                max_idx = rounded_predictions.idxmax()
                rounded_predictions[max_idx] += diff
            adjusted_predictions[non_zero_mask] = rounded_predictions
    return adjusted_predictions


def boost_top_minutes(predictions_df, team_total=240, boost_threshold=36, max_minutes=38):
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()
    for team in predictions_df['TEAM'].unique():
        team_mask = predictions_df['TEAM'] == team
        team_data = adjusted_predictions[team_mask].copy()
        top_5_idx = team_data.nlargest(5).index
        top_5_mins = team_data[top_5_idx]
        eligible_mask = (top_5_mins < boost_threshold) & (top_5_mins > 0)
        if eligible_mask.any():
            available_boost = sum(boost_threshold - mins for mins in top_5_mins[eligible_mask])
            boost_proportions = (boost_threshold - top_5_mins[eligible_mask]) / available_boost
            other_players_idx = team_data.index.difference(top_5_idx)
            if len(other_players_idx) > 0:
                minutes_to_redistribute = min(available_boost * 0.5, team_data[other_players_idx].sum() * 0.01)
                for idx in top_5_mins[eligible_mask].index:
                    boost = minutes_to_redistribute * boost_proportions[idx]
                    adjusted_predictions[idx] = min(adjusted_predictions[idx] + boost, max_minutes)
                if minutes_to_redistribute > 0:
                    reduction_factor = 1 - (minutes_to_redistribute / team_data[other_players_idx].sum())
                    adjusted_predictions[other_players_idx] *= reduction_factor
    return adjusted_predictions


def adjust_team_minutes_with_minimum_and_boost(predictions_df, min_threshold=8, team_total=240, max_minutes=38):
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()
    adjusted_predictions[adjusted_predictions <= min_threshold] = 0
    adjusted_predictions = adjusted_predictions.clip(upper=max_minutes)
    adjusted_predictions = boost_top_minutes(
        predictions_df.assign(Predicted_Minutes=adjusted_predictions),
        max_minutes=max_minutes
    )
    for team in predictions_df['TEAM'].unique():
        team_mask = predictions_df['TEAM'] == team
        team_total_mins = adjusted_predictions[team_mask].sum()
        if team_total_mins > 0:
            non_zero_mask = (predictions_df['TEAM'] == team) & (adjusted_predictions > 0)
            scale_factor = team_total / team_total_mins
            team_predictions = adjusted_predictions[non_zero_mask]
            scaled_predictions = team_predictions * scale_factor
            scaled_predictions = scaled_predictions.clip(upper=max_minutes)
            rounded_predictions = np.round(scaled_predictions, 1)
            total_after_cap = rounded_predictions.sum()
            if total_after_cap < team_total:
                remaining_mins = team_total - total_after_cap
                candidates = rounded_predictions[rounded_predictions < max_minutes].sort_values(ascending=False)
                while remaining_mins > 0 and len(candidates) > 0:
                    for idx in candidates.index:
                        space_to_max = max_minutes - rounded_predictions[idx]
                        if space_to_max > 0:
                            adjustment = min(0.1, remaining_mins, space_to_max)
                            rounded_predictions[idx] += adjustment
                            remaining_mins -= adjustment
                        if remaining_mins <= 0:
                            break
                    if remaining_mins > 0:
                        candidates = rounded_predictions[rounded_predictions < max_minutes].sort_values(ascending=False)
                        if len(candidates) == 0:
                            break
            adjusted_predictions[non_zero_mask] = rounded_predictions
    return adjusted_predictions


def create_advanced_features(df):
    """
    Create advanced features for minutes prediction
    Parameters:
        df: DataFrame containing game logs
    Returns:
        DataFrame with additional advanced features
    """
    # Make sure data is sorted properly
    df = df.sort_values(['PLAYER', 'GAME_DAY'])

    # Initialize new features dictionary
    new_features = {}

    df['Projection'] = df['DK']

    # Momentum and Trend Features
    for window in [3, 5, 7]:
        # Rolling minutes averages
        df[f'MIN_LAST_{window}_AVG'] = df.groupby('PLAYER')['MIN'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Rolling minutes volatility (standard deviation)
        df[f'MIN_LAST_{window}_STD'] = df.groupby('PLAYER')['MIN'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

        # Coefficient of variation (standardized volatility)
        df[f'MIN_LAST_{window}_CV'] = df[f'MIN_LAST_{window}_STD'] / df[f'MIN_LAST_{window}_AVG']

        # Trend (slope of minutes)
        df[f'MIN_TREND_{window}'] = df.groupby('PLAYER')['MIN'].transform(
            lambda x: x.rolling(window).apply(
                lambda y: np.nan if len(y) < 2 else np.polyfit(range(len(y)), y, 1)[0]
            )
        )

    # Role Change Detection
    df['ROLE_CHANGE_3_10'] = (df['MIN_LAST_3_AVG'] - df['MIN_LAST_10_AVG']).abs()
    df['ROLE_CHANGE_5_10'] = (df['MIN_LAST_5_AVG'] - df['MIN_LAST_10_AVG']).abs()

    # Consistency Metrics
    df['MIN_CONSISTENCY_SCORE'] = df.groupby('PLAYER')['MIN'].transform(
        lambda x: x.rolling(10, min_periods=1).std() / x.rolling(10, min_periods=1).mean()
    )

    # Performance-based features
    df['SCORING_EFFICIENCY'] = df['PTS'] / df['MIN']
    df['RECENT_SCORING_EFF'] = df.groupby('PLAYER')['SCORING_EFFICIENCY'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # Game Impact Features
    df['PLUS_MINUS_PER_MIN'] = df['PLUS MINUS'] / df['MIN']
    df['RECENT_IMPACT'] = df.groupby('PLAYER')['PLUS_MINUS_PER_MIN'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # Minutes Distribution Features
    df['MIN_ABOVE_20'] = (df['MIN'] > 20).astype(int)
    df['MIN_ABOVE_25'] = (df['MIN'] > 25).astype(int)
    df['MIN_ABOVE_30'] = (df['MIN'] > 30).astype(int)

    # Rolling percentage of games above thresholds
    for threshold in ['20', '25', '30']:
        df[f'FREQ_ABOVE_{threshold}'] = df.groupby('PLAYER')[f'MIN_ABOVE_{threshold}'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )

    # Add team minutes rank features
    df['TEAM_PROJ_RANK'] = df.groupby(['TEAM', 'GAME_DAY'])['DK_LAST_10_AVG'].rank(ascending=False)
    df['IS_TOP_3_PROJ'] = (df['TEAM_PROJ_RANK'] <= 3).astype(int)

    # Add percentage of team minutes feature
    team_totals = df.groupby(['TEAM', 'GAME_DAY'])['MIN'].transform('sum')
    df['TEAM_MIN_PERCENTAGE'] = (df['MIN'] / team_totals) * 100

    # Create interaction feature for low-minute top players
    df['LOW_MIN_TOP_PLAYER'] = ((df['TEAM_PROJ_RANK'] <= 3) &
                                (df['TEAM_MIN_PERCENTAGE'] < 14)).astype(int)

    # Fill NaN values with appropriate defaults
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    return df


def prepare_training_data(data):
    y = data['MIN'].copy()
    y[y <= 8] = 0
    return y


def save_predictions(all_results, output_dir='predictions'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    consolidated_results = pd.concat(all_results, axis=0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    consolidated_filename = os.path.join(output_dir, f'all_games_minutes_predictions_{timestamp}.csv')
    consolidated_results.to_csv(consolidated_filename, index=False)
    print(f"\nConsolidated results saved to: {consolidated_filename}")


# Load the enhanced dataset
file_path = 'nba_game_logs_enhanced_training_set.csv'
data = pd.read_csv(file_path)

data = create_advanced_features(data)

# Define features
features = [

   # 'MIN_VS_TEAM_AVG',
    'MIN_CUM_AVG',
    'MIN_ABOVE_AVG_STREAK',
    'MIN_LAST_10_AVG',
    'MIN_CONSISTENCY',

     'DK_TREND_5',
     'DK_LAST_10_AVG',
    'PTS_CUM_AVG',
    'REB_PER_MIN',
    'AST_PER_MIN',
    'PTS_PER_MIN',
    'AST_LAST_10_AVG',
    'REB_LAST_10_AVG',

    'DAYS_REST',
    'PTS_LAST_10_AVG',
    #'BLOWOUT_GAME',
    #'IS_HOME',
    'MIN_TREND',
    #'IS_B2B',

    # New advanced features
    'MIN_LAST_3_AVG',
    'MIN_LAST_5_AVG',
    'ROLE_CHANGE_3_10',
    'ROLE_CHANGE_5_10',
    'MIN_CONSISTENCY_SCORE',
    'RECENT_SCORING_EFF',
    'RECENT_IMPACT',

    'FREQ_ABOVE_20',
    'FREQ_ABOVE_25',
    'FREQ_ABOVE_30',

    'TEAM_PROJ_RANK',
    'IS_TOP_3_PROJ',
    'TEAM_MIN_PERCENTAGE',
    'LOW_MIN_TOP_PLAYER',
    'Projection'

]


# Verify all features exist in the dataset
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"Warning: Missing features: {missing_features}")
    # Remove missing features from the feature list
    features = [f for f in features if f in data.columns]
    print(f"Proceeding with available features: {len(features)} features")

# Initialize results storage
results = []
mae_comparison = []
all_results = []

# Iterate through each game day as a test set
for test_game_day in sorted(data['GAME_DAY'].unique()):
    print(f"\nProcessing Game Day: {test_game_day}")

    # Split the data
    train_data = data[data['GAME_DAY'] < test_game_day].copy()
    test_data = data[data['GAME_DAY'] == test_game_day].copy()

    # Skip if not enough training data
    if len(train_data) < 100:
        continue

    X_train = train_data[features]
    y_train = prepare_training_data(train_data)

    # Check for any NaN values
    if X_train.isna().any().any():
        print("Warning: NaN values found in training features")
        X_train = X_train.fillna(0)

    # Train CatBoost model
    cat_model = CatBoostRegressor(
        iterations=250,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=10,
        loss_function='RMSE',
        random_state=42,
        verbose=0,
        #feature_weights=[10.0 if col == 'Projection' else 1 for col in features]
        feature_weights = [
            1 if col == 'TEAM_MIN_PERCENTAGE'
            else 0 if col == 'Projection'
            else 10 if col == 'DK_LAST_10_AVG'
            else 1.0
            for col in features
        ]

    )
    cat_model.fit(X_train, y_train)

    # Make predictions
    X_test = test_data[features]
    if X_test.isna().any().any():
        print("Warning: NaN values found in test features")
        X_test = X_test.fillna(0)

    initial_predictions = cat_model.predict(X_test)
    initial_predictions[initial_predictions <= 8] = 0
    initial_predictions = np.minimum(initial_predictions, 38)

    # Apply adjustments
    test_data['Predicted_Minutes'] = adjust_team_minutes_with_minimum_and_boost(
        test_data.assign(Predicted_Minutes=initial_predictions),
        max_minutes=38
    )

    # Minutes Distribution Analysis
    print("\nMinutes Distribution Analysis:")
    for team in test_data['TEAM'].unique():
        team_data = test_data[test_data['TEAM'] == team]
        print(f"\n{team}:")
        print(f"Team Total: {team_data['Predicted_Minutes'].sum():.1f}")
        #print(f"Max Minutes: {team_data['Predicted_Minutes'].max():.1f}")
        print(f"Players > 35 mins: {len(team_data[team_data['Predicted_Minutes'] > 35])}")
        top_5 = team_data.nlargest(5, 'Predicted_Minutes')
        #print("\nTop 5 Players:")
        #for _, row in top_5.iterrows():
            #print(f"{row['PLAYER']:<20} {row['Predicted_Minutes']:.1f}")

    # Calculate metrics comparing predictions to actual minutes
    predictions = test_data['Predicted_Minutes']
    actual_minutes = test_data['MIN']

    mae = mean_absolute_error(actual_minutes, predictions)
    rmse = np.sqrt(mean_squared_error(actual_minutes, predictions))
    r2 = r2_score(actual_minutes, predictions)
    predicted_corr = pearsonr(actual_minutes, predictions)[0]

    # Store results
    results.append({
        'Game_Day': test_game_day,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 3),
        'Predicted_Correlation': round(predicted_corr, 3)
    })

    mae_comparison.append({
        'Game_Day': test_game_day,
        'Model MAE': round(mae, 2),
        'Original MAE': round(mean_absolute_error(actual_minutes, test_data['MIN_LAST_10_AVG']), 2),
        'Model_Correlation': round(predicted_corr, 3),
        'Original_Correlation': round(pearsonr(actual_minutes, test_data['MIN_LAST_10_AVG'])[0], 3)
    })

    # Store relevant columns for output
    output_columns = [
        'GAME_DAY',
        'PLAYER',
        'TEAM',
        'MIN',  # Actual minutes
        'MIN_LAST_10_AVG',  # Original projection
        'Predicted_Minutes',  # Our prediction
        'DK',
        'PTS',
        'REB',
        'AST'
    ]

    contest_results = test_data[output_columns].copy()
    all_results.append(contest_results)

# Create summary DataFrame
results_df = pd.DataFrame(results)
mae_comparison_df = pd.DataFrame(mae_comparison)

# Calculate and append averages
averages = {
    'Game_Day': 'Average',
    'MAE': results_df['MAE'].mean(),
    'RMSE': results_df['RMSE'].mean(),
    'R2': results_df['R2'].mean(),
    'Predicted_Correlation': results_df['Predicted_Correlation'].mean()
}
results_df = pd.concat([results_df, pd.DataFrame([averages])], ignore_index=True)

# Print results
print("\nDetailed Results by Game Day:")
print(tabulate(results_df, headers='keys', tablefmt='pretty', floatfmt=".3f"))

print("\nModel vs Original Predictions Comparison:")
print(tabulate(mae_comparison_df, headers='keys', tablefmt='pretty', floatfmt=".3f"))

print("\nOverall Correlation Summary:")
print(f"Model Predictions Correlation: {mae_comparison_df['Model_Correlation'].mean():.3f}")
print(f"Original Predictions Correlation: {mae_comparison_df['Original_Correlation'].mean():.3f}")

# Save results
save_predictions(all_results)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nCatBoost Feature Importance:")
print(feature_importance)

# Save the final model
output_file = 'final_minutes_expanded_prediction_model.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(cat_model, f)
print(f"\nModel saved to {output_file}")