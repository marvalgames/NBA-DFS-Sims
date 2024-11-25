import openpyxl
import requests
from bs4 import BeautifulSoup
import pandas as pd
import xlwings as xw
# URL for the player stats page (example: 2023-24 season stats)
url = 'https://www.basketball-reference.com/leagues/NBA_2024_totals.html'

# Request the page content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table and parse
table = soup.find('table', {'id': 'totals_stats'})
rows = table.find_all('tr')

# Extract column headers
headers = [th.text for th in rows[0].find_all('th')][1:]  # Skip the 'Rank' header

# Extract player data
data = []
for row in rows[1:]:
    cols = [td.text for td in row.find_all('td')]
    if cols:  # Skip empty rows
        data.append(cols)

# Create DataFrame

df = pd.DataFrame(data, columns=headers)

# Display first few rows
print(df.head())

# Export DataFrame to CSV
df.to_csv('../output/nba_stats.csv', index=False)
#  '../dk_data/projections.csv'
print("NBA stats exported to 'nba_stats.csv'")


# Define Excel file and sheet name

xlsm_file_path = '../output/nba_dfs.xlsm'
sheet_name = 'Totals'

# Open the workbook and overwrite the sheet
with xw.App(visible=False) as app:  # Use 'visible=True' if you want to see the process in Excel
    wb = app.books.open(xlsm_file_path)
    if sheet_name in [s.name for s in wb.sheets]:
        wb.sheets[sheet_name].clear()  # Clear existing sheet content if sheet exists
    else:
        wb.sheets.add(sheet_name)  # Add sheet if it doesn't exist
    sheet = wb.sheets[sheet_name]

    # Write headers and data
    sheet.range("A1").value = [headers] + data

    # Save and close
    wb.save()
    wb.close()

print(f"Data successfully written to '{sheet_name}' sheet in '{xlsm_file_path}'")

'''
# Define the path to the .xlsm file and the sheet and table names
xlsm_file_path = '../output/nba_dfs.xlsm'
sheet_name = 'Totals'
table_name = 'Totals'  # Replace with the actual table name in Excel

# Open the workbook and access the sheet with xlwings
with xw.App(visible=True) as app:  # Use 'visible=True' to see the updates in real-time
    wb = app.books.open(xlsm_file_path)
    sheet = wb.sheets[sheet_name]

    # Access the Excel table
    excel_table = sheet.api.ListObjects(table_name)

    # Clear current table data (excluding headers)
    excel_table.DataBodyRange.ClearContents()

    # Write headers (if they need updating)
    for col_num, header in enumerate(headers, start=1):
        excel_table.HeaderRowRange.Cells(1, col_num).Value = header

    # Write data to the table
    #for row_num, row_data in enumerate(data, start=1):
    #    for col_num, cell_value in enumerate(row_data, start=1):
    #        excel_table.DataBodyRange.Cells(row_num, col_num).Value = cell_value

    excel_table.DataBodyRange.Resize(len(data), len(data[0])).Value = data

    # Save and close the workbook
    wb.save()
    wb.close()

print(f"Data successfully written to '{table_name}' table in '{sheet_name}' sheet of '{xlsm_file_path}'")
'''

