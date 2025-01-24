import pickle

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os
from tabulate import tabulate

# Load the dataset
file_path = 'nba_minutes.csv'
data = pd.read_csv(file_path)

# Define features for minutes prediction
features = [
    'Salary',
    'Points Proj',
    'Value',
    'Own Actual',
    'Games Played',
    'Plus',
    #'Minutes',  # Original minutes prediction
    'Ceiling'
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
    y_train = train_data['Min Actual']  # Target is actual minutes

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

    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)

    # Add predictions to test data
    test_data['Predicted_Minutes'] = predictions

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
