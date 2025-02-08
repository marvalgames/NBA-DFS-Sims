from lightgbm import LGBMRegressor
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import math
# Load the dataset
file_path = 'dataset_with_feasibility_11_contests.csv'  # Replace with your dataset file path
data = pd.read_csv(file_path)

# Step 1: Adjust ownership for zero-projection players
data.loc[data['Points Proj'] == 0.0, 'Own Actual'] = (
    data['Salary'] / 100000
)


# Step 2: Add Slot Count for Position Eligibility
slot_map = {
    'PG': 3, 'SG': 3, 'SF': 3, 'PF': 3, 'C': 2,
    'PG/SG': 4, 'SG/SF': 5, 'SF/PF': 4, 'PF/C': 4,
    'G': 4, 'F': 4, 'UTIL': 1
}
data['Slot Count'] = data['Position'].map(slot_map).fillna(1).astype(int)

# Step 3: Adjust Feasibility Score
data['Weighted Feasibility'] = data['Feasibility'] * data['Slot Count']

# Step 1: Count the number of players in each contest
data['Player_Pool_Size'] = data.groupby('Contest ID')['Name'].transform('count')


# Interaction Features
data['Proj_Salary_Ratio'] = data['Points Proj'] / data['Salary']   # Points per $1 salary
data['Bonus_Value'] = data['Value'] * data['Plus']     # Points per $1 salary
data['Feasibility_Slot_Interaction'] = data['Feasibility'] * data['Slot Count']  # Feasibility weighted by position eligibility

salary_adjustment = data['Salary'] / 1000 * 10
# data['Value_Boost'] = data['Points Proj'] - salary_adjustment

# Polynomial and Statistical Features
data['Points_Proj_Squared'] = data['Points Proj'] ** 2  # Squared Points Proj
data['Salary_Cubed'] = data['Salary'] ** 3  # Cubed salary

# Aggregated Features
data['Avg_Position_Ownership'] = data.groupby('Position')['Own Actual'].transform('mean')  # Avg ownership by position
data['Salary_Rank'] = data.groupby('Contest ID')['Salary'].rank(ascending=False)
data['Proj_Points_Rank'] = data.groupby('Contest ID')['Points Proj'].rank(ascending=False)
data['Proj_Salary_Interaction'] = data['Points Proj'] * np.log1p(data['Salary'])
data['Value_Boost'] = data['Salary_Rank'] * data['Proj_Points_Rank']

data['Ownership_Trend'] = data.groupby('Name')['Own Actual'].shift(1).rolling(window=3).mean()
data['Ownership_Trend'] = data['Ownership_Trend'].fillna(0)  # Fill NaNs for first games
data['Proj_Value_Interaction'] = data['Points Proj'] * data['Value']
data['Feasibility_Salary_Interaction'] = data['Feasibility'] * (1 / data['Salary'])

data['Position_Share'] = data.groupby(['Contest ID', 'Position'])['Name'].transform('count') / data['Player_Pool_Size']
high_value_threshold = 5.5
data['High_Value_Position_Count'] = data.groupby(['Contest ID', 'Position'])['Value'].transform(
    lambda x: (x >= high_value_threshold).sum()
)


# Step 3: Calculate the ratio of low-salary, high-value players
low_salary_threshold = 4500
high_value_threshold = 5.5

# Identify players meeting the criteria
data['Low_Salary_High_Value'] = (
    (data['Salary'] <= low_salary_threshold) & (data['Value'] >= high_value_threshold)
).astype(int)

# Calculate the proportion of exceptional value players in the contest
data['Low_Salary_High_Value_Ratio'] = data.groupby('Contest ID')['Low_Salary_High_Value'].transform('sum') / data['Player_Pool_Size']

# Boost high-salary players proportionally
amplified_salary_threshold = 9000
salary_ratio = amplified_salary_threshold / data['Salary']
data['Amplified_Value'] = (
    (data['Salary'] > amplified_salary_threshold) * (data['Low_Salary_High_Value_Ratio'] * salary_ratio)
)



# Verify
print(data[['Contest ID', 'Name', 'Salary', 'Amplified_Value']].head(10))


# Export the first 10 rows to CSV
# sample_data = data[['Name', 'Salary', 'Value', 'Amplified_Value', 'Low_Salary_High_Value_Sum', 'High_Salary_Squeeze_Boost', 'Contest ID']]
# sample_data.to_csv("sample_data.csv", index=False)


# Step 4: Train-Test Split
train_data = data[data['Contest ID'] < 11].copy()
test_data = data[data['Contest ID'] == 11].copy()

# Log-transform the target variable
train_data['Log_Own_Actual'] = np.log1p(train_data['Own Actual'])
test_data['Log_Own_Actual'] = np.log1p(test_data['Own Actual'])

# Define features and target
# Updated feature list with interaction features
features = [
    'Salary',
    'Points Proj',
    'Value',
    'Feasibility',
    'Games Played',
    'Plus',
    'Slot Count',
    'Weighted Feasibility',
    'Proj_Salary_Ratio',
    'Points_Proj_Squared',
    'Avg_Position_Ownership',
    'Salary_Rank',
    'Proj_Points_Rank',
    'Proj_Salary_Interaction',
    'Bonus_Value',
    'Value_Boost',
    'Ownership_Trend',
    'Feasibility_Salary_Interaction'
    #'Feasibility_Slot_Interaction',
    #'Amplified_Value',
    #'Salary_Cubed'

]

X_train = train_data[features]
y_train = train_data['Log_Own_Actual']

X_test = test_data[features]
y_test = test_data['Log_Own_Actual']


# LightGBM Regressor
lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
print("Training LightGBM...")
lgbm_model.fit(X_train, y_train)

# Predict and reverse log transformation
lgbm_log_y_pred = lgbm_model.predict(X_test)
lgbm_y_pred = np.expm1(lgbm_log_y_pred)

# Evaluate LightGBM
lgbm_mse = mean_squared_error(np.expm1(y_test), lgbm_y_pred)
lgbm_r2 = r2_score(np.expm1(y_test), lgbm_y_pred)
lgbm_corr = pearsonr(lgbm_y_pred, np.expm1(y_test))[0]

print(f"\nLightGBM Results:")
print(f"Test Mean Squared Error: {lgbm_mse:.4f}")
print(f"Test R^2 Score: {lgbm_r2:.4f}")
print(f"Ownership Correlation: {lgbm_corr:.4f}")


from catboost import CatBoostRegressor

# CatBoost Regressor
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=5,
    loss_function='RMSE',
    random_state=42,
    verbose=100
)
print("Training CatBoost...")
cat_model.fit(X_train, y_train, verbose=500)

# Predict and reverse log transformation
cat_log_y_pred = cat_model.predict(X_test)
cat_y_pred = np.expm1(cat_log_y_pred)

# Evaluate CatBoost
cat_mse = mean_squared_error(np.expm1(y_test), cat_y_pred)
cat_r2 = r2_score(np.expm1(y_test), cat_y_pred)
cat_corr = pearsonr(cat_y_pred, np.expm1(y_test))[0]

print(f"\nCatBoost Results:")
print(f"Test Mean Squared Error: {cat_mse:.4f}")
print(f"Test R^2 Score: {cat_r2:.4f}")
print(f"Ownership Correlation: {cat_corr:.4f}")


# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100,
                                 random_state=42,
                                 max_depth=None, min_samples_split=2, min_samples_leaf=2)



#n_estimators: 100, max_depth: None, min_samples_split: 2, min_samples_leaf: 2, MSE: 18.1151, R^2: 0.7573, Correlation: 0.8770
rf_model.fit(X_train, y_train)

# Predict and reverse log transformation
rf_log_y_pred = rf_model.predict(X_test)
rf_y_pred = np.expm1(rf_log_y_pred)  # Reverse log1p

# Evaluate
rf_mse = mean_squared_error(np.expm1(y_test), rf_y_pred)
rf_r2 = r2_score(np.expm1(y_test), rf_y_pred)
rf_corr = pearsonr(rf_y_pred, np.expm1(y_test))[0]

print(f"\nRandom Forest Results:")
print(f"Test Mean MSE: {rf_mse:.4f}")
print(f"Test R^2: {rf_r2:.4f}")
print(f"Ownership Correlation: {rf_corr:.4f}")

# Enforce the rule for zero-projection players
test_data['Predicted Ownership'] = rf_y_pred
test_data.loc[test_data['Points Proj'] == 0.0, 'Predicted Ownership'] = (
    test_data.loc[test_data['Points Proj'] == 0, 'Salary'] / 100000
)

# Step 7: Evaluate the Adjusted Predictions
# Use the adjusted 'Predicted Ownership' for evaluation
adjusted_y_pred = test_data['Predicted Ownership']

# Recalculate metrics
original_mse = mean_squared_error(np.expm1(y_test), adjusted_y_pred)  # MSE in original scale
r2 = r2_score(np.expm1(y_test), adjusted_y_pred)
correlation = pearsonr(adjusted_y_pred, np.expm1(y_test))[0]

# print(f"\nAdjusted Test Mean Squared Error (Original Scale): {original_mse:.4f}")
# print(f"Adjusted Test R^2 Score: {r2:.4f}")
# print(f"Adjusted Ownership Correlation: {correlation:.4f}")


# Step 8: Save the Trained Model
# ask ChatGpt

# Step 2: Adjust scaling based on contest size and player pool
contest_size = test_data['Contest Size'].iloc[0]  # Contest size (lineups entered)
player_pool_size = test_data['Player_Pool_Size'].iloc[0]  # Number of players in the pool

# Total available slots in the contest
total_slots = contest_size * 8

# Adjust scaling factor to account for the player pool size
scaling_factor = total_slots / player_pool_size  # Average ownership per player

# Apply scaling
test_data['Scaled Ownership'] = (
    test_data['Predicted Ownership'] / test_data['Predicted Ownership'].sum()
) * scaling_factor * player_pool_size

# Convert to percentage
test_data['Scaled Ownership (%)'] = (
    test_data['Scaled Ownership'] / total_slots
) * 800  # Ownership percentage aligned with 8 slots



# Print the sum of Scaled Ownership Percentages
print(f"Sum of Scaled Ownership Percentages: {test_data['Scaled Ownership (%)'].sum():.2f}%")

# Round numeric columns to two decimal places
table = test_data[['Name', 'Own Actual', 'Predicted Ownership', 'Scaled Ownership', 'Scaled Ownership (%)']].head(5)
table = table.round(2)

# Print the table with formatted cells
# print(tabulate(table, headers='keys', tablefmt='pretty', showindex=False))


# Save predictions
output_file = 'test_data_with_predictions.csv'
test_data.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'.")


import matplotlib.pyplot as plt

# Feature Importance for LightGBM
lgbm_importance = lgbm_model.feature_importances_
lgbm_features = pd.DataFrame({'Feature': features, 'Importance': lgbm_importance})
lgbm_features = lgbm_features.sort_values(by='Importance', ascending=False)

# Print and plot
print("\nLightGBM Feature Importance:")
# print(lgbm_features)

# Feature Importance for CatBoost
cat_importance = cat_model.get_feature_importance()
cat_features = pd.DataFrame({'Feature': features, 'Importance': cat_importance})
cat_features = cat_features.sort_values(by='Importance', ascending=False)
# Print and plot
print("\nCatBoost Feature Importance:")
# print(cat_features)

# Feature Importance for Random Forest
rf_importance = rf_model.feature_importances_
rf_features = pd.DataFrame({'Feature': features, 'Importance': rf_importance})
rf_features = rf_features.sort_values(by='Importance', ascending=False)

# Print and plot
print("\nRandom Forest Feature Importance:")
# print(rf_features)


# Export to CSV
lgbm_features.to_csv('lightgbm_feature_importance.csv', index=False)
print("LightGBM feature importance saved to 'lightgbm_feature_importance.csv'")


feature_importance = cat_model.get_feature_importance(prettified=True)
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': cat_model.get_feature_importance()
}).sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)

feature_importance_df.to_csv("catboost_feature_importance.csv", index=False)
print("Feature importance saved to 'catboost_feature_importance.csv'.")



