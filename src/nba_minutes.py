import pickle

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
    """
    Adjust predictions so each team's total minutes equals 48

    Parameters:
    predictions_df: DataFrame containing 'Team' and 'Predicted_Minutes' columns

    Returns:
    Series with adjusted minutes predictions
    """
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


def boost_top_minutes(predictions_df, team_total=240, boost_threshold=36):
    """
    Boost top 5 players' minutes if under 36, maintaining team total of 240
    """
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_data = adjusted_predictions[team_mask].copy()

        # Get top 5 minute players
        top_5_idx = team_data.nlargest(5).index
        top_5_mins = team_data[top_5_idx]

        # Calculate boost for eligible players (under 36 minutes)
        eligible_mask = top_5_mins < boost_threshold
        if eligible_mask.any():
            available_boost = sum(boost_threshold - mins for mins in top_5_mins[eligible_mask])

            # Calculate proportional boost
            boost_proportions = (boost_threshold - top_5_mins[eligible_mask]) / available_boost

            # Calculate minutes to redistribute from non-top 5 players
            other_players_idx = team_data.index.difference(top_5_idx)
            if len(other_players_idx) > 0:
                minutes_to_redistribute = min(available_boost * 0.5,
                                              team_data[
                                                  other_players_idx].sum() * 0.01)  # limit to 10% of other players' minutes

                # Apply boost to eligible top 5 players
                for idx in top_5_mins[eligible_mask].index:
                    boost = minutes_to_redistribute * boost_proportions[idx]
                    adjusted_predictions[idx] += boost

                # Proportionally reduce other players' minutes
                if minutes_to_redistribute > 0:
                    reduction_factor = 1 - (minutes_to_redistribute / team_data[other_players_idx].sum())
                    adjusted_predictions[other_players_idx] *= reduction_factor

    return adjusted_predictions


def adjust_team_minutes_with_minimum_and_boost(predictions_df, min_threshold=8, team_total=240):
    """
    Complete adjustment process including minimum threshold, boost, and rounding
    """
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    # Set predictions under threshold to zero
    adjusted_predictions[adjusted_predictions <= min_threshold] = 0

    # Apply boost to top 5 players
    adjusted_predictions = boost_top_minutes(
        predictions_df.assign(Predicted_Minutes=adjusted_predictions)
    )

    # Final adjustment to maintain team totals
    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_total_mins = adjusted_predictions[team_mask].sum()

        if team_total_mins > 0:
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
                rounded_predictions[max_idx] += round(diff, 1)

            adjusted_predictions[non_zero_mask] = rounded_predictions

    return adjusted_predictions



# In your training loop, modify the target values:
def prepare_training_data(data):
    """Prepare training data with minimum minutes threshold"""
    y = data['Min Actual'].copy()
    # Set actual minutes <= 8 to zero
    y[y <= 8] = 0
    return y

# Load the dataset
file_path = 'nba_minutes.csv'
data = pd.read_csv(file_path)

# Define features for minutes prediction
features = [
    'Salary',
    'Points Proj',
    #'Value',
    #'Own Actual',
    #'Games Played',
    #'Plus',
    #'Minutes',  # Original minutes prediction
    #'Ceiling'
]

# Initialize results storage
results = []
mae_comparison = []

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
    test_data['Predicted_Minutes'] = predictions

    # # Apply adjustments with constraints
    # test_data['Predicted_Minutes'] = adjust_team_minutes_with_constraints(
    #     test_data,
    #     min_minutes=0,
    #     max_minutes=42  # Maximum minutes for any player
    # )

    # Adjust team minutes to sum to 240 while maintaining minimum threshold
    test_data['Predicted_Minutes'] = adjust_team_minutes_with_minimum(test_data)
    predictions = test_data['Predicted_Minutes']


    # Verify results
    team_totals = test_data.groupby('Team')['Predicted_Minutes'].sum()
    print(f"\nTeam Minute Totals for Contest {test_contest_id}:")
    print(team_totals)

    # Verify minimum threshold
    min_mins_count = len(test_data[test_data['Predicted_Minutes'].between(0.1, 8)])
    print(f"\nPredictions between 0-8 minutes: {min_mins_count}")

    zero_mins_count = len(test_data[test_data['Predicted_Minutes'] == 0])
    print(f"Predictions at exactly 0 minutes: {zero_mins_count}")

    team_total_verification = test_data.groupby('Team')['Predicted_Minutes'].sum()
    print("\nTeam Total Verification:")
    print(f"Mean total: {team_total_verification.mean():.1f}")
    print(f"Min total: {team_total_verification.min():.1f}")
    print(f"Max total: {team_total_verification.max():.1f}")

    # In your main loop:
    predictions = cat_model.predict(X_test)
    predictions[predictions <= 8] = 0

    # Apply all adjustments including boost
    test_data['Predicted_Minutes'] = adjust_team_minutes_with_minimum_and_boost(
        test_data.assign(Predicted_Minutes=predictions)
    )

    # Add verification for top 5 players on each team
    print("\nTop 5 Players Minutes by Team:")
    for team in test_data['Team'].unique():
        team_data = test_data[test_data['Team'] == team]
        top_5 = team_data.nlargest(5, 'Predicted_Minutes')
        print(f"\n{team}:")
        print("Player | Minutes")
        print("-" * 30)
        for _, row in top_5.iterrows():
            print(f"{row['DK Name']:<20} {row['Predicted_Minutes']:.1f}")
        print(f"Team Total: {team_data['Predicted_Minutes'].sum():.1f}")


    # # Optional: Smooth the adjustments
    # test_data['Predicted_Minutes'] = smooth_adjustments(
    #     predictions,
    #     test_data['Predicted_Minutes'],
    #     smoothing_factor=0.7
    # )
    predictions = test_data['Predicted_Minutes']

    # # Final adjustment to ensure exactly 48 minutes per team
    # test_data['Predicted_Minutes'] = adjust_team_minutes(test_data)

    # Calculate metrics
    mae = mean_absolute_error(test_data['Min Actual'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['Min Actual'], predictions))
    r2 = r2_score(test_data['Min Actual'], predictions)

    # Calculate correlations
    predicted_corr = pearsonr(test_data['Min Actual'], predictions)[0]
    original_corr = pearsonr(test_data['Min Actual'], test_data['Minutes'])[0]



    # Save detailed results for each contest
    output_file = os.path.join('predictions', f'minutes_predictions_{test_contest_id}.csv')
    results_df = test_data[['DK Name', 'Team', 'Position', 'Min Actual', 'Predicted_Minutes'] + features]
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'")



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
