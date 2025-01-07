import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
from scipy.optimize import minimize



# Post-process predictions to apply the 1% ownership cap for low minutes players
def apply_low_minutes_cap(predictions, data):
    capped_predictions = predictions.copy()
    capped_predictions[data['Low_Minutes_Flag'] == 1] = np.minimum(capped_predictions[data['Low_Minutes_Flag'] == 1], 0.01)
    return capped_predictions

# Load the dataset
file_path = 'dataset_with_feasibility_weighted.csv'
data = pd.read_csv(file_path)

# Step 2: Add Slot Count for Position Eligibility
slot_map = {
    'PG': 3, 'SG': 3, 'SF': 3, 'PF': 3, 'C': 2,
    'PG/SG': 4, 'SG/SF': 5, 'SF/PF': 4, 'PF/C': 4, 'PG/SF': 5,
    'G': 4, 'F': 4, 'UTIL': 1
}

top_n = 40  # Change this value as needed
metric_field = 'Points Proj'  # Change to your desired metric field
data['Team_Contest_Rank'] = data.groupby(['Contest ID', 'Team'])[metric_field].rank(ascending=False, method='dense')
#data.loc[data['Team_Contest_Rank'] >= top_n, 'Points Proj'] = 0
#data.loc[data['Team_Contest_Rank'] >= top_n, 'Value'] = 0



salary = data['Salary']
minutes = data['Minutes']
points = data['Points Proj']
value = data['Value']
ceiling = data['Ceiling']
max_points = points + ceiling
feasibility = data['Feasibility']
games_played = data['Games Played']
ceiling_value = ceiling / salary
#feasibility = 1


data['Slot Count'] = data['Position'].map(slot_map).fillna(1).astype(int)
data['Player_Pool_Size'] = data.groupby('Contest ID')['DK Name'].transform('nunique')
data['Feasibility'] = feasibility * minutes
#data['Weighted Feasibility'] = data['Feasibility'] * (data['Slot Count'])
data['Proj_Salary_Ratio'] = points / salary
data['Normalized_Value'] = data.groupby('Contest ID')['Value'].transform(lambda x: (x - x.mean()) / x.std())
data['Log_Proj'] = np.log1p(data['Points Proj'])
data['Prev_Salary_Mean'] = data.groupby('DK Name')['Salary'].shift(1).rolling(window=3).mean()
#data['Ceiling'] = (data['Points Proj'] + data['Ceiling'])
data['Salary_Proj_Interaction'] = data['Salary'] * data['Points Proj']
data['Value_Feasibility_Interaction'] = data['Value'] * data['Feasibility']
data['Proj_Feasibility_Interaction'] = data['Points Proj'] * data['Feasibility']
data['Low_Minutes_Flag'] = (data['Minutes'] <= 12).astype(int)
data['Low_Minutes_Penalty'] = data['Low_Minutes_Flag'] * data['Points Proj']
data['High_Points_Boost'] = ((data['Points Proj'] >= 50) & (data['Value'] >= 5.5)).astype(int)

data['Team_Total_Points_Proj'] = data.groupby('Team')['Points Proj'].transform('sum')
data['Player_Points_Percentage'] = data['Points Proj'] / data['Team_Total_Points_Proj']

data['Value_Plus'] = points - salary / 1000 * 4
data['Ceiling_Plus'] = max_points - salary / 1000 * 5


shift_start = 0.4
shift_end = 1.0
add = 1.0  # Constant to avoid log issues with zero
drift = 1.35
players_per_game = 48  # Number of players per game to rank for the shift

features = [
        'Tournament Feasibility',
        'Player_Points_Percentage',
        'Salary',
        'Points Proj',
        'Proj_Salary_Ratio',
        'Value',
        'Feasibility',
        'Weighted Feasibility',
        'Prev_Salary_Mean',
        'Log_Proj',
        'Salary_Proj_Interaction',
        'Proj_Feasibility_Interaction',
        'Value_Feasibility_Interaction',
        'Normalized_Value',
        'Ceiling',
        'Low_Minutes_Penalty',
        'Value_Plus',
        'Ceiling_Plus',
]


import numpy as np
# Parameters for dynamic shift

# Function to compute dynamic shift
def compute_dynamic_shift(rank, total, shift_start, shift_end):
    scale = (rank / total) ** drift
    return shift_start + (shift_end - shift_start) * scale

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Variable for the ownership threshold
min_ownership_threshold = 0.0  # Minimum 1% ownership to include in SD calculation

from catboost import CatBoostRegressor, Pool
from catboost.utils import eval_metric
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import os
from tabulate import tabulate

# Function to calculate a custom metric
def custom_metric(y_true, y_pred):
    """
    Example custom metric: Penalize predictions > 100
    """
    diff = y_pred - y_true
    penalty = np.maximum(0, y_pred - 100)  # Penalize predictions over 100
    return np.mean(diff**2 + penalty**2)


# Iterate through each contest as a test set
results = []
sd_comparison = []  # To store SD results for debugging
alpha =  .99# Weight for tournament_feasibility
data['Custom_Target'] = alpha * data['Tournament Feasibility'] + (1 - alpha) * data['Own Actual']

for test_contest_id in data['Contest ID'].unique():
    print(f"\nProcessing Contest ID: {test_contest_id}")

    def rescale_with_bounds(predictions: pd.Series, target_sd=7.3229, min_bound=0.0):
        numbers = predictions.values
        index = predictions.index
        original_mean = np.mean(numbers)

        # Split into small (<1) and large values but scale both
        small_mask = (numbers < 1)
        large_mask = ~small_mask

        # Get reference ranges for maintaining proportions of small values
        small_values = numbers[small_mask]
        if len(small_values) > 0:
            small_min = np.min(small_values)
            small_max = np.max(small_values)

        def objective(params):

            result = numbers.copy()

            # Scale large values
            if np.any(large_mask):
                large_centered = numbers[large_mask] - np.mean(numbers[large_mask])
                result[large_mask] = large_centered * params[0] + np.mean(numbers[large_mask])

            # Scale small values while preserving order
            if np.any(small_mask):
                small_scaled = (numbers[small_mask] - small_min) / (small_max - small_min)
                small_scaled = small_scaled * params[1] + params[2]
                result[small_mask] = small_scaled

            current_sd = np.std(result)
            current_mean = np.mean(result)

            # Penalties
            sd_penalty = (current_sd - target_sd) ** 2 * 3000
            mean_penalty = (current_mean - original_mean) ** 2 * 800
            min_penalty = np.sum(np.maximum(0, min_bound - result) ** 2) * 1000

            # Penalty for small values order violation
            order_penalty = 0
            if np.any(small_mask):
                small_diffs = np.diff(result[small_mask])
                order_penalty = np.sum(np.maximum(0, -small_diffs)) * 2000

            return sd_penalty + mean_penalty + min_penalty + order_penalty

        # Initial guess
        initial_guess = [
            target_sd / np.std(numbers),  # large values scale
            0.5,  # small values scale
            min_bound  # small values shift
        ]

        # Optimize
        result = minimize(objective, initial_guess, method='Nelder-Mead',
                          options={'maxiter': 2000})

        # Apply final transformation
        final_result = numbers.copy()

        # Apply to large values
        if np.any(large_mask):
            large_centered = numbers[large_mask] - np.mean(numbers[large_mask])
            final_result[large_mask] = large_centered * result.x[0] + np.mean(numbers[large_mask])

        # Apply to small values
        if np.any(small_mask):
            small_scaled = (numbers[small_mask] - small_min) / (small_max - small_min)
            small_scaled = small_scaled * result.x[1] + result.x[2]
            final_result[small_mask] = small_scaled

        # Ensure minimum bound
        final_result = np.maximum(final_result, min_bound)

        # Return as pandas Series with original index
        return pd.Series(final_result, index=index)



    def predict_sd(players):
        # SD = 0.0006 × Players^2 −0.1509 × Players + 19.3296
        a = 0.0001  # Coefficient for Players^2
        b = -0.0770  # Coefficient for Players
        c = 21.0  # Intercept
        players = np.array(players)  # Ensure the input is a NumPy array
        return a * (players ** 2) + b * players + c


    # Split the data
    train_data = data[data['Contest ID'] != test_contest_id].copy()
    unfiltered_test_data = data[data['Contest ID'] == test_contest_id].copy()  # Keep unfiltered data for saving
    test_data = unfiltered_test_data.copy()  # Create a copy for filtering

    # Rank players and filter top-ranked players
    train_data['Rank'] = train_data.groupby('Contest ID')['Own Actual'].rank(ascending=False)
    test_data['Rank'] = test_data.groupby('Contest ID')['Own Actual'].rank(ascending=False)

    rank_limit_train = train_data['Games Played'].iloc[0] * 48
    rank_limit_test = test_data['Games Played'].iloc[0] * 48

    train_data = train_data[train_data['Rank'] <= rank_limit_train]
    test_data = test_data[test_data['Rank'] <= rank_limit_test]

    X_train = train_data[features]
    X_test = test_data[features]


    y_train = (train_data['Custom_Target'] + add) ** shift_start
    #y_train = (train_data['Own Actual'] + add) ** shift_start
    y_test = (test_data['Own Actual'] + add) ** shift_start

    # Train CatBoost model
    cat_model = CatBoostRegressor(
        iterations=250,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=12,
        loss_function='RMSE',
        random_state=42,
        verbose=0,
        feature_weights=[1 if col == 'Tournament Feasibility' else 1 for col in features]
    )
    cat_model.fit(X_train, y_train)

    # Generate preliminary predictions for ranking
    preliminary_predictions = cat_model.predict(X_test)
    test_data['Preliminary_Predictions'] = (preliminary_predictions ** (1 / shift_start)) - add

    # Scale preliminary predictions to sum to 800
    scaling_factor = 800 / test_data['Preliminary_Predictions'].sum()
    test_data['Scaled_Predictions'] = test_data['Preliminary_Predictions'] * scaling_factor

    rank_limit = test_data['Games Played'].iloc[0] * players_per_game
    test_data['Group_Rank'] = test_data['Scaled_Predictions'].rank(ascending=False)

    test_data['Dynamic_Shift'] = test_data.apply(lambda row: compute_dynamic_shift(row['Group_Rank'], rank_limit, shift_start, shift_end)
        if row['Group_Rank'] <= rank_limit else shift_end, axis=1)

    y_test_dynamic = (test_data['Own Actual'] + add) ** test_data['Dynamic_Shift']
    y_train_dynamic = (train_data['Custom_Target'] + add) ** shift_start
    #y_train_dynamic = (train_data['Own Actual'] + add) ** shift_start
    cat_model.fit(X_train, y_train_dynamic)

    # Final predictions
    log_predictions = cat_model.predict(X_test)
    predictions = (log_predictions ** (1 / test_data['Dynamic_Shift'])) - add

    # Scale predictions to sum to 800
    scaling_factor = 800 / predictions.sum()
    predictions *= scaling_factor

    # Apply low-minutes cap and adjustments
    test_data['Predicted Ownership'] = apply_low_minutes_cap(predictions, test_data)

    test_data.loc[test_data['Points Proj'] == 0.0, 'Predicted Ownership'] = (
            test_data.loc[test_data['Points Proj'] == 0, 'Salary'] / 1000000
    )

    test_data['Predicted Ownership'] = test_data['Predicted Ownership'].clip(lower=0)
    predictions = test_data['Predicted Ownership']

    current_sum = predictions.sum()

    player_pool = len(test_data)
    estimated_sd = predict_sd(player_pool)
    estimated_sd_multiplier = 0.935 * (current_sum / 800.0)  # Set your desired SD value
    target_sd = estimated_sd * estimated_sd_multiplier

    target_sum = 800  # Ensure predictions sum to 800
    print(f'Players : {player_pool} Sum: {current_sum} ')
    print(f'Estimated SD: {estimated_sd} Multiplier: {estimated_sd_multiplier} ')
    print(f'Target SD: {target_sd}')

    #adjusted_predictions = adjust_sd(predictions, target_sd, target_sum)
    adjusted_predictions = rescale_with_bounds(predictions, target_sd=target_sd)
    final_predictions = apply_low_minutes_cap(adjusted_predictions, test_data)
    test_data['Predicted Ownership'] = final_predictions
    threshold = 1e-6
    test_data.loc[test_data['Points Proj'] < threshold, 'Predicted Ownership'] = (test_data.loc[test_data['Points Proj'] == 0, 'Salary'] / 1000000)
    test_data['Predicted Ownership'] = test_data['Predicted Ownership'].clip(lower=0)

    final_predictions = test_data['Predicted Ownership']
    scaling_factor = 800 / final_predictions.sum()
    final_predictions *= scaling_factor
    test_data['Predicted Ownership'] = final_predictions

    # Debugging: Print before scaling and final results
    print(f"Before Scaling - Mean: {predictions.mean():.4f}, SD: {predictions.std():.4f} Total: {predictions.sum()}")
    print(f"After Scaling Adjusted - Mean: {adjusted_predictions.mean():.4f}, SD: {adjusted_predictions.std():.4f}, Total: {adjusted_predictions.sum():.4f}")
    print(f"After Scaling Final - Mean: {final_predictions.mean():.4f}, SD: {final_predictions.std():.4f}, Total: {final_predictions.sum():.4f}")


    output_file = os.path.join('..', 'training', f'all_players_results_{test_contest_id}.csv')
    test_data.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'.")


    # Calculate SDs
    own_actual_sd = test_data['Own Actual'].std()
    predicted_ownership_sd = test_data['Predicted Ownership'].std()

    # Save SDs for debugging
    sd_comparison.append({
        'Contest ID': test_contest_id,
        'Own Actual SD': round(own_actual_sd, 3),
        'Predicted Ownership SD': round(predicted_ownership_sd, 3)
    })

    # Reverse transformation on y_test for evaluation
    reverse_y_test = (y_test ** (1 / test_data['Dynamic_Shift'])) - add
    #reverse_y_test = y_test

    # Evaluate metrics on the original scale
    cat_mse = mean_squared_error(reverse_y_test, final_predictions)
    cat_r2 = r2_score(reverse_y_test, final_predictions)
    cat_corr = pearsonr(final_predictions, reverse_y_test)[0]

    print(f"Feasibility Total: {test_data['Feasibility'].sum():.2f}")
    print(f"Scaled Ownership Total: {test_data['Predicted Ownership'].sum():.2f}")
    print(f"Predicted Ownership Total: {final_predictions.sum():.2f}")
    print(f"Predicted Ownership SD: {final_predictions.std():.2f}\n")
    print(test_data[['DK Name', 'Predicted Ownership']].head(3))

    # Create a summary DataFrame
    results_df = pd.DataFrame(test_data)

    # Save results to a CSV file
    # Step 2: Load the Excel sheet
    output_file = os.path.join('..', 'training', f'all_players_results_{test_contest_id}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'.")

    # Save results
    results.append({
        'Contest ID': test_contest_id,
        'Player Pool': len(test_data),
        'Test MSE': round(cat_mse, 3),
        'Test R^2': round(cat_r2, 3),
        'Ownership Correlation': round(cat_corr, 3)
    })


    # Group by 'Contest ID' and calculate the total 'Predicted Ownership' for each contest
    predicted_ownership_totals = test_data.groupby('Contest ID')['Predicted Ownership'].sum().reset_index()

    # Rename columns for clarity
    predicted_ownership_totals.columns = ['Contest ID', 'Total Predicted Ownership']

    # Round the totals to two decimal places for neatness
    predicted_ownership_totals['Total Predicted Ownership'] = predicted_ownership_totals[
        'Total Predicted Ownership'].round(2)

    # Print the results as a table
    print("\nTotal Predicted Ownership by Contest:")
    print(tabulate(predicted_ownership_totals, headers='keys', tablefmt='pretty', floatfmt=".2f"))


# Display SD comparison
sd_comparison_df = pd.DataFrame(sd_comparison)
print("\nStandard Deviation Comparison by Contest:")
print(tabulate(sd_comparison_df, headers='keys', tablefmt='pretty', floatfmt=".3f"))

# Create a summary DataFrame
results_df = pd.DataFrame(results)

# Calculate averages
averages = {
    'Contest ID': 'Average',
    'Player Pool': round(results_df['Player Pool'].mean(), 1),
    'Test MSE': round(results_df['Test MSE'].mean(), 3),
    'Test R^2': round(results_df['Test R^2'].mean(), 3),
    'Ownership Correlation': round(results_df['Ownership Correlation'].mean(),3)
}

# Append averages row to the DataFrame
results_df = pd.concat([results_df, pd.DataFrame([averages])], ignore_index=True)

# Print results with rounded values
print("\nSummary of Results:")
print(tabulate(results_df, headers='keys', tablefmt='pretty', floatfmt=".2f"))

# Save results to a CSV file
output_file = 'all_contests_results.csv'
results_df.to_csv(output_file, index=False)
print(f"Results saved to '{output_file}'.")



# Save the model
output_file = 'final_nba_model.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(cat_model, f)
print(f"Model saved to {output_file}.")


# Get feature importance
feature_importance = pd.DataFrame({
   'feature': features,
   'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

# Print the feature importance
print("CatBoost Feature Importance:")
print(feature_importance)



