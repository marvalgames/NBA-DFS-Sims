import sys
from pathlib import Path
import os
import pandas as pd
import xlwings as xw
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget


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

        fta_button = QPushButton("Import fta.csv to SOG Projections")
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

    def import_dk_entries(self):
        print("Importing DKEntries to dk_list...")
        self.import_csv_to_sheet(
            csv_file="entries.csv",
            sheet_name="dk_list",
            csv_start_row=7,
            csv_start_col=0,
            excel_start_row=2,
            excel_start_col=2,
        )
        print("Done importing DKEntries to dk_list.")

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
        print("Importing Advanced to team_ratings...")
        self.import_csv_to_sheet(
            csv_file="advanced.csv",
            sheet_name="team_ratings",
            csv_start_row=0,
            csv_start_col=1,
            excel_start_row=2,
            excel_start_col=2,
        )
        print("Done importing Advanced to team_ratings.")

    def import_traditional(self):
        print("Importing Traditional to team_ratings...")
        self.import_csv_to_sheet(
            csv_file="traditional.csv",
            sheet_name="team_ratings",
            csv_start_row=0,
            csv_start_col=1,
            excel_start_row=2,
            excel_start_col=25,
        )
        print("Done importing Advanced to team_ratings.")

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
            self, csv_file, sheet_name, csv_start_row, csv_start_col, excel_start_row, excel_start_col
    ):
        # Define the current script folder and sibling folder
        #current_folder = Path(__file__).parent  # Current script directory (src)
        #target_folder = current_folder.parent/"dk-import"  # Sibling folder (dk-import)

        # Always combine target_folder with the csv_file parameter
        #combined_csv_file = target_folder/Path(csv_file).name
        combined_csv_file = csv_file

        # Excel file is in the same target_folder
        #excel_file = target_folder/"nba.xlsm"
        excel_file = "nba.xlsm"

        print(f"Excel File: {excel_file}")
        print(f"Combined CSV File: {combined_csv_file}")

        # Open Excel and write data
        app = xw.App(visible=False)  # Set to True to see Excel
        try:
            wb = app.books.open(excel_file)

            # Load CSV data
            data = pd.read_csv(combined_csv_file)
            data_to_write = data.iloc[csv_start_row:, csv_start_col:]

            # Write data to the specified sheet
            ws = wb.sheets[sheet_name]
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
