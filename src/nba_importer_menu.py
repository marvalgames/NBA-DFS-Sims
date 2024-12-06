import sys
from pathlib import Path
import os
import pandas as pd
import xlwings as xw
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from nba_api.stats.endpoints import leaguedashteamstats

from nba_fetch import fetch_advanced_team_stats


class ImportTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel Import Tool")
        self.setGeometry(100, 100, 400, 300)

        # Combine all styles into a single setStyleSheet call
        self.setStyleSheet("""
                           QMainWindow {
                               background-color: #2E3440;  /* Dark background */
                               color: white;               /* Default text color */
                           }

                           QPushButton {
                               background-color: #5E81AC;
                               color: white;
                               font-size: 16px;
                               font-weight: bold;
                               border-radius: 10px;
                               min-height: 30px;
                           }
                           QPushButton:hover {
                               background-color: #81A1C1;
                           }

                           QLineEdit {
                               background-color: #3B4252;  /* Dark background */
                               color: white;               /* White text */
                               border: 1px solid #555555;  /* Border color */
                               border-radius: 5px;         /* Rounded corners */
                               padding: 5px;               /* Inner padding */
                               font-size: 16px;            /* Font size */
                               font-family: Arial;         /* Font name */
                               font-weight: bold;          /* Font weight (bold, normal, etc.) */
                           }

                           QLineEdit:focus {
                               border: 1px solid #88C0D0;  /* Highlighted border on focus */
                               background-color: #434C5E;  /* Slightly lighter background */
                           }

                           QComboBox {
                               background-color: #3B4252;  /* Dark background */
                               color: white;               /* White text */
                               border: 1px solid #555555;  /* Border color */
                               border-radius: 5px;         /* Rounded corners */
                               padding: 5px;               /* Inner padding */
                               font-size: 14px;            /* Font size */
                           }

                             QSpinBox {
                               width: 64px;
                               background-color: #3B4252;  /* Dark background */
                               color: white;               /* White text */
                               border: 1px solid #555555;  /* Border color */
                               border-radius: 5px;         /* Rounded corners */
                               padding: 5px;               /* Inner padding */
                               font-size: 14px;            /* Font size */
                               font-weight: bold;          /* Font weight (bold, normal, etc.) */
                           }

                           QSpinBox::up-button, QSpinBox::down-button {
                               width: 32px;  /* Wider button area for arrows */
                               height: 16px; /* Taller button area for arrows */
                           }

                            QSpinBox::up-arrow {
                               width: 32px;  /* Increase arrow width */
                               height: 16px; /* Increase arrow height */
                           }

                           QSpinBox::down-arrow {
                               width: 32px;  /* Increase arrow width */
                               height: 16px; /* Increase arrow height */
                           }

                           QComboBox::drop-down {
                               subcontrol-origin: padding;
                               subcontrol-position: top right;
                               width: 40px;
                               border-left: 1px solid #555555;
                               background-color: #2E3440;  /* Dropdown arrow background */
                           }

                           QComboBox QAbstractItemView {
                               background-color: #3B4252;  /* Dropdown list background */
                               color: white;               /* Text color */
                               selection-background-color: #88C0D0; /* Selected item background */
                               selection-color: black;     /* Selected item text color */
                           }

                           QLabel {
                               color: #88C0D0;  /* Default text color */
                               font-size: 16px;
                               font-weight: bold;
                           }
                       """)


        # Define the target folder
        current_folder = Path(__file__).parent  # Current script directory (src)
        target_folder = current_folder.parent / "dk_import"  # Sibling folder (dk-import)

        # Change the working directory
        os.chdir(target_folder)

        # Verify the working directory
        print(f"Current working directory: {os.getcwd()}")

        # Layout
        layout = QVBoxLayout()
        # Buttons

        fta_button = QPushButton("Import fta.csv to FTA Projections")
        fta_button.clicked.connect(self.import_fta_entries)

        dk_button = QPushButton("Import entries.csv to DK List")
        dk_button.clicked.connect(self.import_dk_entries)

        sog_button = QPushButton("Import entries.csv to SOG Projections")
        sog_button.clicked.connect(self.import_sog_projections)

        darko_button = QPushButton("Import darko.csv to DARKO Projections")
        darko_button.clicked.connect(self.import_darko)

        advanced_button = QPushButton("Import advanced.csv to Advanced Team Stats")
        advanced_button.clicked.connect(self.import_advanced)

        traditional_button = QPushButton("Import traditional.csv to Traditional Team Stats")
        traditional_button.clicked.connect(self.import_traditional)

        all_button = QPushButton("Run All Imports")
        all_button.clicked.connect(self.run_all_imports)

        # Add buttons to layout
        layout.addWidget(fta_button)
        layout.addWidget(dk_button)
        layout.addWidget(sog_button)
        layout.addWidget(darko_button)
        layout.addWidget(advanced_button)
        layout.addWidget(traditional_button)
        layout.addWidget(all_button)

        # Central Widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def import_fta_entries(self):
        print("Importing NBA-DK-Data to dk_list...")
        self.import_csv_to_sheet(
            csv_file="fta.csv",
            sheet_name="fta",
            csv_start_row=0,
            csv_start_col=0,
            excel_start_row=2,
            excel_start_col=1,
        )
        print("Done importing FTA to fta.")

    import os
    from pathlib import Path
    import pandas as pd
    import xlwings as xw

    def import_dk_entries(self):
        print("Importing DKEntries to dk_list...")
        csv_file = "entries.csv"
        excel_file = "nba.xlsm"
        sheet_name = "dk_list"

        # Define the output directory and CSV file path
        current_folder = Path(__file__).resolve().parent
        sibling_folder = current_folder.parent / "dk_data"  # Sibling folder
        sibling_folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist
        output_csv_file = sibling_folder / "DKSalaries.csv"  # Output CSV file path

        try:
            # Step 1: Define the columns to read
            columns_to_read = list(range(13, 22))  # Columns N (13) to V (21) in zero-based indexing

            # Step 2: Load the CSV
            data = pd.read_csv(
                csv_file,
                skiprows=7,  # Skip the first 7 rows (rows 1-7)
                usecols=columns_to_read,  # Only read columns N–V
                header=0  # Use the first remaining row (row 8) as headers
            )

            # Step 3: Write the same data to the new CSV file
            data.to_csv(output_csv_file, index=False)
            print(f"Data successfully written to '{output_csv_file}'.")

            # Step 4: Open Excel and write the data
            app = xw.App(visible=False)  # Set to True if you want to see the Excel window
            try:
                wb = app.books.open(excel_file)  # Open the Excel workbook
                ws = wb.sheets[sheet_name]  # Select the target sheet

                # Clear the range in the sheet before writing
                ws.range("B2").expand().clear()  # Clear existing data starting from column B, row 2

                # Write the data to the sheet
                ws.range("B2").value = data.values

                # Save the workbook
                wb.save()
                print(f"Data successfully imported to sheet '{sheet_name}' in '{excel_file}'.")
            finally:
                wb.close()
                app.quit()
        except Exception as e:
            print(f"An error occurred: {e}")

        print("Done importing DKEntries to dk_list.")

    def import_sog_projections(self):
        print("Importing DKEntries to dk_list...")
        csv_file = "entries.csv"
        excel_file = "nba.xlsm"
        sheet_name = "sog_projections"
        try:
            # Step 1: Define the columns to read
            columns_to_read = list(range(13, 22))  # Columns N (13) to V (21) in zero-based indexing

            # Step 2: Load the CSV
            data = pd.read_csv(
                csv_file,
                skiprows=7,  # Skip the first 7 rows (rows 1-7)
                usecols=columns_to_read,  # Only read columns N–V
                header=0  # Use the first remaining row (row 8) as headers
            )

            # Step 3: Open Excel and write the data
            app = xw.App(visible=False)  # Set to True if you want to see the Excel window
            try:
                wb = app.books.open(excel_file)  # Open the Excel workbook
                ws = wb.sheets[sheet_name]  # Select the target sheet

                # Define the target range for writing data (10 columns starting from column B)
                #target_range = ws.range("B2").resize(data.shape[0], 10)

                # Overwrite only the necessary range
                #target_range.value = data.values

                # Clear the range in the sheet before writing
                #ws.range("B2").expand().clear()  # Clear existing data starting from column N, row 2

                # Write the data to the sheet
                ws.range("B2").resize(data.shape[0], 10)
                ws.range("B2").value = data.values

                # Save the workbook
                wb.save()
                print(f"Data successfully imported to sheet '{sheet_name}' in '{excel_file}'.")
            finally:
                wb.close()
                app.quit()
        except Exception as e:
            print(f"An error occurred: {e}")


        #self.import_csv_to_sheet(
          #  csv_file="entries.csv",
          #  sheet_name="dk_list",
           # csv_start_row=7,
           # csv_start_col=0,
           # excel_start_row=2,
           # excel_start_col=2,
        #)


        print("Done importing DKEntries to dk_list.")
    '''
    def import_sog_projections(self):
        print("Importing DKEntries to sog_projections...")
        self.import_csv_to_sheet(
            csv_file="entries.csv",
            sheet_name="sog_projections",
            csv_start_row=7,
            csv_start_col=0,
            excel_start_row=2,
            excel_start_col=2,
        )
        print("Done importing DKEntries to sog_projections.")
    '''

    def import_darko(self):
        print("Importing DARKO to darko...")
        self.import_csv_to_sheet(
            csv_file="darko.csv",
            sheet_name="darko",
            csv_start_row=0,
            csv_start_col=0,
            excel_start_row=2,
            excel_start_col=1,
        )
        print("Done importing DARKO to darko.")

    def import_advanced(self):
        print("Fetching and importing Advanced Team Stats...")

        try:
            # Step 1: Fetch the data
            season = '2024-25'
            season_type = 'Regular Season'

            # Fetch the data using the function
            df = fetch_advanced_team_stats(season, season_type)

            # Sort the DataFrame by 'TEAM_NAME'
            df_sorted = df.sort_values(by='TEAM_NAME')
            csv_file_path = "advanced.csv"

            # Save the DataFrame to the CSV
            df_sorted.to_csv(csv_file_path, index=False)
            print(f"Advanced stats saved to: {csv_file_path}")

            # Step 3: Import the saved CSV to the "team_ratings" sheet in the Excel file

            # Specify the columns to read
            columns_to_read = [0, 1, 2, 3, 4, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 25]

            self.import_csv_to_sheet(
                csv_file=str(csv_file_path),
                sheet_name="team_ratings",
                csv_start_row=0,
                csv_start_col=0,  # No need to filter columns here; `usecols` handles it
                excel_start_row=2,
                excel_start_col=1,
                usecols=columns_to_read
            )

            print("Done importing Advanced stats to team_ratings.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def import_traditional(self):
        print("Fetching and importing Traditional Team Stats...")

        try:
            # Step 1: Fetch the data
            season = '2024-25'
            season_type = 'Regular Season'

            # Fetch the data using nba_api's endpoint for traditional stats
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star=season_type,
                measure_type_detailed_defense='Base',  # 'Base' corresponds to traditional stats
                per_mode_detailed='Totals'  # Totals per mode
            )

            # Convert to DataFrame
            df = team_stats.get_data_frames()[0]

            # Sort the DataFrame by 'TEAM_NAME'
            df_sorted = df.sort_values(by='TEAM_NAME')

            # Step 2: Save the fetched data to 'traditional.csv' in the output folder
            #script_dir = Path(__file__).resolve().parent  # Script directory
            #output_dir = script_dir / "output"  # Define the output folder
            #output_dir.mkdir(exist_ok=True)  # Create the folder if it doesn't exist

            # Define the CSV file path
            #csv_file_path = output_dir / "traditional.csv"
            csv_file_path = "traditional.csv"

            # Save the DataFrame to the CSV
            df_sorted.to_csv(csv_file_path, index=False)
            print(f"Traditional stats saved to: {csv_file_path}")

            # Step 3: Import the saved CSV to the "team_ratings" sheet in the Excel file

            # Specify the columns to read
            columns_to_read = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]  # columns to import

            self.import_csv_to_sheet(
                csv_file=str(csv_file_path),
                sheet_name="team_ratings",
                csv_start_row=0,
                csv_start_col=0,  # No need to filter columns here; `usecols` handles it
                excel_start_row=2,
                excel_start_col=24,  # Start writing from a different column
                usecols=columns_to_read
            )

            print("Done importing Traditional stats to team_ratings.")
        except Exception as e:
            print(f"An error occurred: {e}")


    def run_all_imports(self):
        print("Running all imports...")
        self.import_fta_entries()
        self.import_dk_entries()
        self.import_sog_projections()
        self.import_darko()
        self.import_advanced()
        self.import_traditional()
        print("All imports completed.")

    from pathlib import Path
    import pandas as pd
    import xlwings as xw

    def import_csv_to_sheet(
            self, csv_file, sheet_name, csv_start_row, csv_start_col, excel_start_row, excel_start_col, usecols=None
    ):
        # Combine the CSV file path
        combined_csv_file = csv_file

        # Excel file
        excel_file = "nba.xlsm"

        print(f"Excel File: {excel_file}")
        print(f"Combined CSV File: {combined_csv_file}")

        # Open Excel and write data
        app = xw.App(visible=False)  # Set to True to see Excel
        try:
            wb = app.books.open(excel_file)

            # Load CSV data with optional column filtering
            data = pd.read_csv(combined_csv_file, usecols=usecols)
            data_to_write = data.iloc[csv_start_row:, csv_start_col:]
            csv_row_count, csv_col_count = data_to_write.shape

            # Write data to the specified sheet
            ws = wb.sheets[sheet_name]

            # Clear the existing rows in the target sheet (up to the CSV's column count)
            clear_range = ws.range(
                (excel_start_row, excel_start_col),
                (excel_start_row + csv_row_count + 1000, excel_start_col + csv_col_count - 1)
            )
            clear_range.value = None  # Clear the cells

            # Write the new data
            ws.range((excel_start_row, excel_start_col)).value = data_to_write.values

            # Save and close the workbook
            wb.save()
        finally:
            wb.close()
            app.quit()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImportTool()
    window.show()
    sys.exit(app.exec())
