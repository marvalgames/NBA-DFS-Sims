import pickle
import pandas as pd
import numpy as np
import xlwings as xw
import os


def adjust_team_minutes_with_minimum_and_boost(predictions_df, min_threshold=8, team_total=240, max_minutes=38):
    adjusted_predictions = predictions_df['Predicted_Minutes'].copy()

    # Set predictions under threshold to zero and cap at max_minutes
    adjusted_predictions[adjusted_predictions <= min_threshold] = 0
    adjusted_predictions = adjusted_predictions.clip(upper=max_minutes)

    # Adjust team totals to 240
    for team in predictions_df['Team'].unique():
        team_mask = predictions_df['Team'] == team
        team_total_mins = adjusted_predictions[team_mask].sum()

        if team_total_mins > 0:
            non_zero_mask = (predictions_df['Team'] == team) & (adjusted_predictions > 0)
            scale_factor = team_total / team_total_mins
            team_predictions = adjusted_predictions[non_zero_mask]
            scaled_predictions = team_predictions * scale_factor
            scaled_predictions = scaled_predictions.clip(upper=max_minutes)
            rounded_predictions = np.round(scaled_predictions, 1)

            # Final adjustment to maintain team total
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


def predict_minutes():
    # Load the trained model
    with open('final_minutes_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    app = xw.App(visible=False)
    try:
        # Connect to Excel
        excel_path = os.path.join('..', 'dk_import', 'nba_min.xlsm')
        wb = xw.Book(excel_path)
        sheet = wb.sheets['sog_minutes']

        # Get the last row with actual data
        last_row = sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row

        # Read only the columns we need for prediction
        needed_data = {
            'Player': sheet.range(f'A2:A{last_row}').value,
            'Team': sheet.range(f'B2:B{last_row}').value,
            'Salary': sheet.range(f'D2:D{last_row}').value,
            'Minutes': sheet.range(f'E2:E{last_row}').value,
            'Projection': sheet.range(f'M2:M{last_row}').value,
        }

        # Convert to DataFrame
        data = pd.DataFrame(needed_data)

        # Filter out rows where Team is None or blank
        data = data.dropna(subset=['Team'])
        data = data[data['Team'] != '']

        # Convert numeric columns
        data['Salary'] = pd.to_numeric(data['Salary'], errors='coerce')
        data['Minutes'] = pd.to_numeric(data['Minutes'], errors='coerce')
        data['Projection'] = pd.to_numeric(data['Projection'], errors='coerce')

        # Create DK feature from Projection
        data['DK'] = data['Projection']
        data['DK Name'] = data['Player']


        # Set Minutes to 0 where Projection is 0
        data.loc[data['DK'] == 0, 'Minutes'] = 0


        # Features used in training
        features = ['Salary', 'DK', 'Minutes']

        # Make predictions
        X = data[features]
        predictions = model.predict(X)

        # Apply adjustments
        data['Predicted_Minutes'] = adjust_team_minutes_with_minimum_and_boost(
            data.assign(Predicted_Minutes=predictions)
        )

        # Create a mapping of predictions
        predictions_dict = dict(zip(data['Player'], data['Predicted_Minutes']))

        # Create predictions array matching original structure
        predictions_to_write = [predictions_dict.get(player, '') if player else ''
                                for player in needed_data['Player']]

        # Write predictions to SOG Minutes column (column H) vertically
        sheet.range(f'H2').options(transpose=True).value = predictions_to_write

        # Print summary by team
        print("\nFinal Predictions:")
        for team in sorted(data['Team'].unique()):
            team_data = data[data['Team'] == team]
            print(f"\n{team}:")
            print(f"Team Total: {team_data['Predicted_Minutes'].sum():.1f}")
            print("\nTop Players:")
            top_players = team_data.nlargest(8, 'Predicted_Minutes')
            for _, row in top_players.iterrows():
                print(f"{row['DK Name']:<20} {row['Predicted_Minutes']:.1f}")

        # Save and close
        wb.save()
        wb.close()

    finally:
        # Make sure to quit the Excel application
        app.quit()


if __name__ == "__main__":
    predict_minutes()