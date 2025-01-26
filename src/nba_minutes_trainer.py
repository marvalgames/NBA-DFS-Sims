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
    """
    Smooth the transition between original and adjusted predictions
    """
    return (original_predictions * (1 - smoothing_factor) +
            adjusted_predictions * smoothing_factor)



def adjust_team_minutes(predictions_df):
    #Adjust predictions so each team's total minutes equals 48

    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_total = adjusted_predictions[team_mask].sum()

        if team_total > 0:  # Avoid division by zero
            # Scale factor to adjust to 48 minutes
            scale_factor = 240 / team_total
            adjusted_predictions[team_mask] = adjusted_predictions[team_mask] * scale_factor

    return adjusted_predictions


def adjust_team_minutes_with_constraints(predictions_df, min_minutes=0, max_minutes=48):
    """
    Adjust predictions with individual player constraints
    """
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_predictions = adjusted_predictions[team_mask]
        team_total = team_predictions.sum()

        if team_total > 0:
            # Initial scaling
            scale_factor = 240 / team_total
            scaled_predictions = team_predictions * scale_factor

            # Apply individual constraints
            scaled_predictions = scaled_predictions.clip(min_minutes, max_minutes)

            # Re-adjust to ensure sum is exactly 48
            final_scale = 240 / scaled_predictions.sum()
            adjusted_predictions[team_mask] = scaled_predictions * final_scale

    return adjusted_predictions


def adjust_team_minutes_with_minimum(predictions_df, min_threshold=8, team_total=240):
    """
    Adjust predictions with minimum minutes threshold and team total of 240
    """
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    # Set predictions under threshold to zero
    adjusted_predictions[adjusted_predictions <= min_threshold] = 0
    # Adjust team totals to 240
    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_total_mins = adjusted_predictions[team_mask].sum()

        if team_total_mins > 0:  # Avoid division by zero
            # Scale non-zero predictions to sum to 240
            non_zero_mask = (predictions_df['Team'] == team) & (adjusted_predictions > 0)
            scale_factor = team_total / team_total_mins
            adjusted_predictions[non_zero_mask] *= scale_factor

            # Round to 1 decimal place while maintaining team total
            team_predictions = adjusted_predictions[non_zero_mask]
            rounded_predictions = np.round(team_predictions, 1)

            # Adjust largest minute prediction to ensure team total is exactly 240
            rounded_total = rounded_predictions.sum()
            if rounded_total != team_total:
                diff = team_total - rounded_total
                max_idx = rounded_predictions.idxmax()
                rounded_predictions[max_idx] += diff

            adjusted_predictions[non_zero_mask] = rounded_predictions

    return adjusted_predictions


def boost_top_minutes(predictions_df, team_total=240, boost_threshold=36, max_minutes=38):
    """
    Boost top 5 players' minutes if under 36, maintaining team total of 240,
    with maximum of 38 minutes
    """
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_data = adjusted_predictions[team_mask].copy()

        # Get top 5 minute players
        top_5_idx = team_data.nlargest(5).index
        top_5_mins = team_data[top_5_idx]

        # Calculate boost for eligible players (under 36 minutes)
        eligible_mask = (top_5_mins < boost_threshold) & (top_5_mins > 0)
        if eligible_mask.any():
            available_boost = sum(boost_threshold - mins for mins in top_5_mins[eligible_mask])

            # Calculate proportional boost
            boost_proportions = (boost_threshold - top_5_mins[eligible_mask]) / available_boost

            # Calculate minutes to redistribute from non-top 5 players
            other_players_idx = team_data.index.difference(top_5_idx)
            if len(other_players_idx) > 0:
                minutes_to_redistribute = min(available_boost * 0.5,
                                              team_data[other_players_idx].sum() * 0.01)

                # Apply boost to eligible top 5 players
                for idx in top_5_mins[eligible_mask].index:
                    boost = minutes_to_redistribute * boost_proportions[idx]
                    adjusted_predictions[idx] = min(adjusted_predictions[idx] + boost, max_minutes)

                # Proportionally reduce other players' minutes
                if minutes_to_redistribute > 0:
                    reduction_factor = 1 - (minutes_to_redistribute / team_data[other_players_idx].sum())
                    adjusted_predictions[other_players_idx] *= reduction_factor

    return adjusted_predictions


def adjust_team_minutes_with_minimum_and_boost(predictions_df, min_threshold=8, team_total=240, max_minutes=38):
    """
    Complete adjustment process including minimum threshold, boost, and rounding
    with maximum of 38 minutes
    """
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    # Set predictions under threshold to zero and cap at max_minutes
    adjusted_predictions[adjusted_predictions <= min_threshold] = 0
    adjusted_predictions = adjusted_predictions.clip(upper=max_minutes)

    # Apply boost to top 5 players
    adjusted_predictions = boost_top_minutes(
        predictions_df.assign(Predicted_Minutes=adjusted_predictions),
        max_minutes=max_minutes
    )

    # Final adjustment to maintain team totals
    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_total_mins = adjusted_predictions[team_mask].sum()

        if team_total_mins > 0:
            non_zero_mask = (predictions_df['Team'] == team) & (adjusted_predictions > 0)
            scale_factor = team_total / team_total_mins

            # Apply scaling while respecting max_minutes
            team_predictions = adjusted_predictions[non_zero_mask]
            scaled_predictions = team_predictions * scale_factor

            # Cap at max_minutes after scaling
            scaled_predictions = scaled_predictions.clip(upper=max_minutes)

            # Round to 1 decimal place
            rounded_predictions = np.round(scaled_predictions, 1)

            # Adjust to maintain team total after max_minutes cap
            total_after_cap = rounded_predictions.sum()
            if total_after_cap < team_total:
                # Distribute remaining minutes to highest minute players under max_minutes
                remaining_mins = team_total - total_after_cap
                candidates = rounded_predictions[rounded_predictions < max_minutes].sort_values(ascending=False)

                while remaining_mins > 0 and len(candidates) > 0:
                    for idx in candidates.index:
                        space_to_max = max_minutes - rounded_predictions[idx]
                        if space_to_max > 0:
                            adjustment = min(0.1, remaining_mins, space_to_max)  # Add in 0.1 minute increments
                            rounded_predictions[idx] += adjustment
                            remaining_mins -= adjustment

                        if remaining_mins <= 0:
                            break

                    if remaining_mins > 0:
                        # If still have minutes to distribute, update candidates
                        candidates = rounded_predictions[rounded_predictions < max_minutes].sort_values(ascending=False)
                        if len(candidates) == 0:
                            # If no more candidates under max_minutes, break
                            break

            adjusted_predictions[non_zero_mask] = rounded_predictions

    return adjusted_predictions





# In your training loop, modify the target values:
def prepare_training_data(data):
    """Prepare training data with minimum minutes threshold"""
    y = data['Min Actual'].copy()
    # Set actual minutes <= 8 to zero
    y[y <= 8] = 0
    return y


def save_predictions(all_results, output_dir='predictions'):
    """
    Save predictions to both individual contest files and a consolidated file

    Parameters:
    all_results: List of DataFrames containing results for each contest
    output_dir: Directory to save the files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Combine all results
    consolidated_results = pd.concat(all_results, axis=0)

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save consolidated file
    consolidated_filename = os.path.join(output_dir, f'all_contests_minutes_predictions_{timestamp}.csv')
    consolidated_results.to_csv(consolidated_filename, index=False)
    print(f"\nConsolidated results saved to: {consolidated_filename}")

    # Save individual contest files
    for contest_data in all_results:
        contest_id = contest_data['Contest ID'].iloc[0]
        filename = os.path.join(output_dir, f'contest_{contest_id}_minutes_predictions.csv')
        contest_data.to_csv(filename, index=False)
        #print(f"Contest {contest_id} results saved to: {filename}")

# Load the dataset
file_path = 'nba_minutes_1.csv'
data = pd.read_csv(file_path)

# Define features for minutes prediction
features = [
    'Salary',
    #'Points Proj',
    'DK',
    #'Plus Minus'
    #'Value',
    #'Own Actual',
    #'Games Played',
    #'Plus',
    'Minutes',  # Original minutes prediction
    #'Ceiling'
]

# Initialize results storage
results = []
mae_comparison = []

# In your main loop:
all_results = []  # List to store results from all contests
# Iterate through each contest as a test set
for test_contest_id in data['Contest ID'].unique():
    print(f"\nProcessing Contest ID: {test_contest_id}")

    # Split the data
    train_data = data[data['Contest ID'] != test_contest_id].copy()
    test_data = data[data['Contest ID'] == test_contest_id].copy()

    X_train = train_data[features]
    #y_train = train_data['Min Actual']  # Target is actual minutes
    y_train = prepare_training_data(train_data)  # Apply minimum threshold to training data

    # Train CatBoost model
    cat_model = CatBoostRegressor(
        iterations=250,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=10,
        loss_function='RMSE',
        random_state=42,
        verbose=0
    )
    cat_model.fit(X_train, y_train)

    # Make predictions
    X_test = test_data[features]
    predictions = cat_model.predict(X_test)
    # In your main loop:
    predictions = cat_model.predict(X_test)
    predictions[predictions <= 8] = 0
    predictions = np.minimum(predictions, 38)  # Apply max minutes cap

    # Apply all adjustments including boost and max minutes
    test_data['Predicted_Minutes'] = adjust_team_minutes_with_minimum_and_boost(
        test_data.assign(Predicted_Minutes=predictions),
        max_minutes=38
    )

    # Add verification for minutes constraints
    print("\nMinutes Distribution Analysis:")
    for team in test_data['Team'].unique():
        team_data = test_data[test_data['Team'] == team]
        print(f"\n{team}:")
        print(f"Team Total: {team_data['Predicted_Minutes'].sum():.1f}")
        print(f"Max Minutes: {team_data['Predicted_Minutes'].max():.1f}")
        print(f"Players > 35 mins: {len(team_data[team_data['Predicted_Minutes'] > 35])}")
        top_5 = team_data.nlargest(5, 'Predicted_Minutes')
        print("\nTop 5 Players:")
        for _, row in top_5.iterrows():
            print(f"{row['DK Name']:<20} {row['Predicted_Minutes']:.1f}")

    predictions = test_data['Predicted_Minutes']

    # Calculate metrics
    mae = mean_absolute_error(test_data['Min Actual'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['Min Actual'], predictions))
    r2 = r2_score(test_data['Min Actual'], predictions)

    # Calculate correlations
    predicted_corr = pearsonr(test_data['Min Actual'], predictions)[0]
    original_corr = pearsonr(test_data['Min Actual'], test_data['Minutes'])[0]



    # Compare with original minutes prediction
    original_mae = mean_absolute_error(test_data['Min Actual'], test_data['Minutes'])


    # Save results with correlations
    results.append({
        'Contest ID': test_contest_id,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 3),
        'Predicted_Correlation': round(predicted_corr, 3)
    })

    mae_comparison.append({
        'Contest ID': test_contest_id,
        'Model MAE': round(mae, 2),
        'Original Prediction MAE': round(original_mae, 2),
        'Model_Correlation': round(predicted_corr, 3),
        'Original_Correlation': round(original_corr, 3)
    })

    # Store relevant columns for output
    output_columns = [
        'Contest ID',
        'DK Name',
        'Team',
        'Position',
        'Salary',
        'Points Proj',
        'Minutes',  # Original projection
        'Predicted_Minutes',  # Our prediction
        'Min Actual',  # Actual minutes (if available)
        'Value',
        'Plus',
        'Ceiling'
        # Add any other columns you want to keep
    ]

    contest_results = test_data[output_columns].copy()

    # Add prediction metrics for this contest
    team_totals = contest_results.groupby('Team')['Predicted_Minutes'].sum()
    max_minutes = contest_results.groupby('Team')['Predicted_Minutes'].max()

    # Add metadata about predictions
    contest_results['Team_Total_Minutes'] = contest_results['Team'].map(team_totals)
    contest_results['Team_Max_Minutes'] = contest_results['Team'].map(max_minutes)

    all_results.append(contest_results)

    # Calculate and display metrics
    print(f"\nContest {test_contest_id} Summary:")
    print(f"Total Players: {len(contest_results)}")
    print(f"Average Team Total: {team_totals.mean():.1f}")
    print(f"Average Team Max Minutes: {max_minutes.mean():.1f}")


# Create summary DataFrame
results_df = pd.DataFrame(results)
mae_comparison_df = pd.DataFrame(mae_comparison)

# Calculate and append averages with correlations
averages = {
    'Contest ID': 'Average',
    'MAE': results_df['MAE'].mean(),
    'RMSE': results_df['RMSE'].mean(),
    'R2': results_df['R2'].mean(),
    'Predicted_Correlation': results_df['Predicted_Correlation'].mean()
}
results_df = pd.concat([results_df, pd.DataFrame([averages])], ignore_index=True)

# Print results with correlations
print("\nDetailed Results by Contest:")
print(tabulate(results_df, headers='keys', tablefmt='pretty', floatfmt=".3f"))

print("\nModel vs Original Predictions Comparison:")
print(tabulate(mae_comparison_df, headers='keys', tablefmt='pretty', floatfmt=".3f"))

# Add overall correlation summary
print("\nOverall Correlation Summary:")
print(f"Model Predictions Correlation: {mae_comparison_df['Model_Correlation'].mean():.3f}")
print(f"Original Predictions Correlation: {mae_comparison_df['Original_Correlation'].mean():.3f}")


# After all contests are processed, save results
save_predictions(all_results)


# Get feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nCatBoost Feature Importance:")
print(feature_importance)

# Save the final model
output_file = 'final_minutes_prediction_model.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(cat_model, f)
print(f"\nModel saved to {output_file}")
