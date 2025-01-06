import xlwings as xw
import pandas as pd
import pickle
import os

print(f"Current working directory: {os.getcwd()}")

# Step 1: Load the trained model
with open('final_xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Step 2: Set the file path and open workbook in hidden mode
file_path = os.path.join('..', 'dk_import', 'nba.xlsm')  # Navigate to sibling directory
app = xw.App(visible=False)  # Create an invisible Excel application
wb = app.books.open(file_path)  # Open the workbook invisibly
sheet = wb.sheets["ownership"]  # Replace with the correct sheet name

# Step 3: Read the data
data = sheet.range('A1').expand().value  # Read the entire table

# Step 4: Stop reading when the 'Name' column is 0 or ""
filtered_data = []
for row in data:
    if row[0] == 0 or row[0] == "" or row[0] is None:  # Check the first column ('Name')
        break
    filtered_data.append(row)

# Step 5: Convert to a Pandas DataFrame
columns = filtered_data[0]  # Extract header row
rows = filtered_data[1:]    # Extract the rest of the rows
df = pd.DataFrame(rows, columns=columns)

# Step 6: Ensure the "Ownership" column exists
if 'Ownership' not in columns:
    raise ValueError("The 'Ownership' column does not exist in the sheet!")

# Step 7: Calculate Games Played dynamically
unique_teams = df['Team'].nunique()  # Count unique teams
games_played = unique_teams // 2     # Divide by 2 to get the number of games
df['Games Played'] = games_played    # Add a 'Games Played' column

# Step 8: Keep only required features for prediction
required_features = ['Points Proj', 'Value', 'FP / Min', 'Salary', 'Proj Min', 'Plus']  # Exclude 'Games Played'
predict_df = df[required_features + ['Games Played']]

# Step 9: Generate predictions
df['Ownership'] = xgb_model.predict(predict_df)  # Add the predictions to the DataFrame

# Step 10: Write predictions back to the existing "Ownership" column
# Find the column index for "Ownership"
ownership_col_index = columns.index('Ownership') + 1  # 1-based index for xlwings
ownership_range = sheet.range(2, ownership_col_index)  # Start from row 2

# Write predictions to the "Ownership" column
ownership_range.value = [[value] for value in df['Ownership']]

# Save and close the workbook
wb.save()
wb.close()
app.quit()

print("Predictions written to the existing 'Ownership' column successfully!")




