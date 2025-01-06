from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# Load the dataset
file_path = 'dataset_with_feasibility_11_contests.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Step 1: Adjust ownership for zero-projection players
data.loc[data['Points Proj'] == 0.0, 'Own Actual'] = data['Salary'] / 100000

# Step 2: Add Slot Count and Adjust Feasibility
slot_map = {
    'PG': 3, 'SG': 3, 'SF': 3, 'PF': 3, 'C': 2,
    'PG/SG': 4, 'SG/SF': 5, 'SF/PF': 4, 'PF/C': 4,
    'G': 4, 'F': 4, 'UTIL': 1
}
data['Slot Count'] = data['Position'].map(slot_map).fillna(1).astype(int)
data['Weighted Feasibility'] = data['Feasibility'] * data['Slot Count']
data['Player_Pool_Size'] = data.groupby('Contest ID')['Name'].transform('count')

# Interaction Features
data['Proj_Salary_Ratio'] = data['Points Proj'] / data['Salary']
data['Bonus_Value'] = data['Value'] * data['Plus']
data['Feasibility_Slot_Interaction'] = data['Feasibility'] * data['Slot Count']

# Train-Test Split
train_data = data[data['Contest ID'] < 11].copy()
test_data = data[data['Contest ID'] == 11].copy()

# Log-transform the target variable
train_data['Log_Own_Actual'] = np.log1p(train_data['Own Actual'])
test_data['Log_Own_Actual'] = np.log1p(test_data['Own Actual'])

# Feature list
features = [
    'Salary', 'Points Proj', 'Value', 'Feasibility', 'Games Played', 'Plus',
    'Slot Count', 'Weighted Feasibility', 'Proj_Salary_Ratio',
    'Feasibility_Slot_Interaction', 'Bonus_Value'
]
X_train = train_data[features]
y_train = train_data['Log_Own_Actual']
X_test = test_data[features]
y_test = test_data['Log_Own_Actual']

# Hyperparameter Tuning
results = []
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 8, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for min_samples_split in param_grid['min_samples_split']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                # Train Random Forest
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Predict and reverse log transformation
                log_y_pred = model.predict(X_test)
                y_pred = np.expm1(log_y_pred)

                # Evaluate the model
                mse = mean_squared_error(np.expm1(y_test), y_pred)
                r2 = r2_score(np.expm1(y_test), y_pred)
                corr = pearsonr(y_pred, np.expm1(y_test))[0]

                # Store results
                results.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'MSE': mse,
                    'R^2': r2,
                    'Correlation': corr
                })

                print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, "
                      f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, "
                      f"MSE: {mse:.4f}, R^2: {r2:.4f}, Correlation: {corr:.4f}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('random_forest_tuning_results.csv', index=False)

# Print T
