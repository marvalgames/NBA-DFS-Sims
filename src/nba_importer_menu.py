import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

import numpy as np
import pandas as pd
import requests
import xlwings as xw
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox, \
    QTextEdit, QProgressDialog
from nba_api.stats.endpoints import leaguedashteamstats

#from nba_fetch import fetch_advanced_team_stats


# In your MainApp class, add these imports at the top:
from PyQt6.QtCore import QThread, pyqtSignal, Qt


class ImportThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            # Override print function for the importer
            original_print = print

            def progress_print(*args):
                message = ' '.join(map(str, args))
                self.progress.emit(message)
                original_print(*args)

            # Add print function to kwargs
            self.kwargs['progress_print'] = progress_print

            # Run the actual import function
            self.func(*self.args, **self.kwargs)
            self.finished.emit(True, "Import completed successfully!")
        except Exception as e:
            self.finished.emit(False, str(e))


class ImportTool(QMainWindow):
    def __init__(self):
        super().__init__()

        # Layout
        layout = QVBoxLayout()
        # Buttons

        bbm_button = QPushButton("Import bbm.csv to BBM Projections")
        bbm_button.clicked.connect(lambda: self.run_threaded_import(self.import_bbm, "Importing BBM data..."))

        fta_button = QPushButton("Import fta.csv to FTA Projections")
        fta_button.clicked.connect(lambda: self.run_threaded_import(self.import_fta_entries, "Importing FTA data..."))

        dk_button = QPushButton("Import entries.csv to DK List")
        dk_button.clicked.connect(lambda: self.run_threaded_import(self.import_dk_entries, "Importing DK entries..."))

        sog_button = QPushButton("Import entries.csv to SOG Projections")
        sog_button.clicked.connect(
            lambda: self.run_threaded_import(self.import_sog_projections, "Importing SOG projections..."))

        darko_button = QPushButton("Import darko.csv to DARKO Projections")
        darko_button.clicked.connect(lambda: self.run_threaded_import(self.import_darko, "Importing DARKO data..."))

        last10_button = QPushButton("Import last10.csv to L10 Sheet")
        last10_button.clicked.connect(
            lambda: self.run_threaded_import(self.import_last10, "Importing last 10 games data..."))

        all_button = QPushButton("Run All Imports")
        all_button.clicked.connect(
            lambda: self.run_threaded_import(self.run_all_imports, "Running all imports..."))

        advanced_button = QPushButton("Import advanced.csv to Advanced Team Stats Sheet")
        advanced_button.clicked.connect(lambda: self.run_threaded_import(
            self.import_advanced, "Importing Advanced Stats..."))

        traditional_button = QPushButton("Import traditional.csv to Traditional Team Stats Sheet")
        traditional_button.clicked.connect(lambda: self.run_threaded_import(
            self.import_traditional, "Importing Traditional Stats..."))

        odds_button = QPushButton("Export NBA Game Odds to NBA Sheet")
        odds_button.clicked.connect(lambda: self.run_threaded_import(
            self.fetch_and_save_team_data_with_odds, "Fetching Game Odds..."))

        export_button = QPushButton("Export Point Projections to NBA Sheet")
        export_button.clicked.connect(lambda: self.run_threaded_import(
            self.export_projections, "Exporting Projections..."))

        own_button = QPushButton("Export Ownership Projections to NBA Sheet")
        own_button.clicked.connect(lambda: self.run_threaded_import(
            self.ownership_projections, "Calculating ownership projections..."))



        quit_button = QPushButton("Quit", self)
        quit_button.clicked.connect(self.close)



        # Add buttons to layout
        layout.addWidget(bbm_button)
        layout.addWidget(fta_button)
        layout.addWidget(dk_button)
        layout.addWidget(sog_button)
        layout.addWidget(darko_button)
        layout.addWidget(advanced_button)
        layout.addWidget(traditional_button)
        layout.addWidget(last10_button)
        layout.addWidget(all_button)

        layout.addWidget(odds_button)
        layout.addWidget(own_button)
        layout.addWidget(export_button)

        layout.addWidget(quit_button)


        # Central Widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Add progress display
        self.progress_display = QTextEdit(self)
        self.progress_display.setReadOnly(True)
        layout.addWidget(self.progress_display)  # Add to your layout appropriately

    def show_progress_dialog(self, title):
        self.progress_dialog = QProgressDialog(title, None, 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setFixedWidth(400)
        self.progress_dialog.setMinimumHeight(160)
        self.progress_dialog.setWindowTitle('Running')

        label = self.progress_dialog.findChild(QLabel)
        if label:
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.progress_dialog.show()

    def update_progress(self, message):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setLabelText(message.strip())
        self.progress_display.append(message)
        QApplication.processEvents()

    def import_finished(self, success, message):
        self.progress_dialog.close()
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", f"An error occurred: {message}")

    def run_threaded_import(self, func, title, *args, **kwargs):
        try:
            self.show_progress_dialog(title)
            self.import_thread = ImportThread(func, *args, **kwargs)
            self.import_thread.progress.connect(self.update_progress)
            self.import_thread.finished.connect(self.import_finished)
            self.import_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

        self.setWindowTitle("Excel Import Tool")
        #self.setGeometry(200, 200, 400, 300)
        # Combine all styles into a single setStyleSheet call
        # self.setStyleSheet("""
        #                    QMainWindow {
        #                        background-color: #2E3440;  /* Dark background */
        #                        color: white;               /* Default text color */
        #                    }
        #
        #                    QPushButton {
        #                        background-color: #5E81AC;
        #                        color: white;
        #                        font-size: 16px;
        #                        font-weight: regular;
        #                        border-radius: 10px;
        #                        min-height: 30px;
        #                    }
        #                    QPushButton:hover {
        #                        background-color: #81A1C1;
        #                    }
        #
        #                    QLineEdit {
        #                        background-color: #3B4252;  /* Dark background */
        #                        color: white;               /* White text */
        #                        border: 1px solid #555555;  /* Border color */
        #                        border-radius: 5px;         /* Rounded corners */
        #                        padding: 5px;               /* Inner padding */
        #                        font-size: 16px;            /* Font size */
        #                        font-family: Arial;         /* Font name */
        #                        font-weight: bold;          /* Font weight (bold, normal, etc.) */
        #                    }
        #
        #                    QLineEdit:focus {
        #                        border: 1px solid #88C0D0;  /* Highlighted border on focus */
        #                        background-color: #434C5E;  /* Slightly lighter background */
        #                    }
        #
        #                    QComboBox {
        #                        background-color: #3B4252;  /* Dark background */
        #                        color: white;               /* White text */
        #                        border: 1px solid #555555;  /* Border color */
        #                        border-radius: 5px;         /* Rounded corners */
        #                        padding: 5px;               /* Inner padding */
        #                        font-size: 14px;            /* Font size */
        #                    }
        #
        #                      QSpinBox {
        #                        width: 64px;
        #                        background-color: #3B4252;  /* Dark background */
        #                        color: white;               /* White text */
        #                        border: 1px solid #555555;  /* Border color */
        #                        border-radius: 5px;         /* Rounded corners */
        #                        padding: 5px;               /* Inner padding */
        #                        font-size: 14px;            /* Font size */
        #                        font-weight: bold;          /* Font weight (bold, normal, etc.) */
        #                    }
        #
        #                    QSpinBox::up-button, QSpinBox::down-button {
        #                        width: 32px;  /* Wider button area for arrows */
        #                        height: 16px; /* Taller button area for arrows */
        #                    }
        #
        #                     QSpinBox::up-arrow {
        #                        width: 32px;  /* Increase arrow width */
        #                        height: 16px; /* Increase arrow height */
        #                    }
        #
        #                    QSpinBox::down-arrow {
        #                        width: 32px;  /* Increase arrow width */
        #                        height: 16px; /* Increase arrow height */
        #                    }
        #
        #                    QComboBox::drop-down {
        #                        subcontrol-origin: padding;
        #                        subcontrol-position: top right;
        #                        width: 40px;
        #                        border-left: 1px solid #555555;
        #                        background-color: #2E3440;  /* Dropdown arrow background */
        #                    }
        #
        #                    QComboBox QAbstractItemView {
        #                        background-color: #3B4252;  /* Dropdown list background */
        #                        color: white;               /* Text color */
        #                        selection-background-color: #88C0D0; /* Selected item background */
        #                        selection-color: black;     /* Selected item text color */
        #                    }
        #
        #                    QLabel {
        #                        color: #88C0D0;  /* Default text color */
        #                        font-size: 16px;
        #                        font-weight: bold;
        #                    }
        #                """)


        # Define the target folder
        current_folder = Path(__file__).parent  # Current script directory (src)
        target_folder = current_folder.parent / "dk_import"  # Sibling folder (dk-import)

        # Change the working directory
        os.chdir(target_folder)

        # Verify the working directory
        print(f"Current working directory: {os.getcwd()}")


        #self.ownership_projections()
        #self.fetch_and_save_team_data_with_odds()
        #self.import_last10()

    # def run_bbm_import(self):
    #     self.run_threaded_import(self.import_bbm, "Importing BBM data...")

    def import_csv_to_sheet(self, csv_file, sheet_name, csv_start_row, csv_start_col,
                            excel_start_row, excel_start_col, usecols=None, progress_print=print):
        app = None
        wb = None

        try:
            # Verify CSV file exists
            if not Path(csv_file).exists():
                raise FileNotFoundError(f"CSV file not found: {csv_file}")

            progress_print(f"Reading CSV file: {csv_file}")
            excel_file = "nba.xlsm"

            # Verify Excel file exists
            if not Path(excel_file).exists():
                raise FileNotFoundError(f"Excel file not found: {excel_file}")

            # Load CSV data first to ensure it's valid before opening Excel
            progress_print("Loading CSV data...")
            data = pd.read_csv(csv_file, usecols=usecols)
            if data.empty:
                raise ValueError("No data found in CSV file")

            data_to_write = data.iloc[csv_start_row:, csv_start_col:]
            csv_row_count, csv_col_count = data_to_write.shape

            if csv_row_count == 0 or csv_col_count == 0:
                raise ValueError("No data to write after applying row/column filters")

            # Initialize Excel
            progress_print("Opening Excel application...")
            app = xw.App(visible=False)
            app.display_alerts = False
            app.screen_updating = False

            if app is None:
                raise RuntimeError("Failed to initialize Excel application")

            progress_print("Opening Excel workbook...")
            wb = app.books.open(excel_file)

            if wb is None:
                raise RuntimeError(f"Failed to open workbook: {excel_file}")

            # Get worksheet and verify it exists
            try:
                ws = wb.sheets[sheet_name]
            except Exception as e:
                raise ValueError(f"Sheet '{sheet_name}' not found in workbook") from e

            progress_print(f"Writing {csv_row_count} rows to sheet {sheet_name}...")

            # Verify the clear range is valid
            progress_print("Clearing existing data...")
            try:
                clear_range = ws.range(
                    (excel_start_row, excel_start_col),
                    (excel_start_row + csv_row_count + 1000, excel_start_col + csv_col_count - 1)
                )
                if clear_range is None:
                    raise ValueError("Failed to create clear range")
                clear_range.clear_contents()
            except Exception as e:
                raise ValueError(f"Error clearing range: {str(e)}")

            # Verify the write range is valid
            progress_print("Writing new data...")
            try:
                write_range = ws.range((excel_start_row, excel_start_col))
                if write_range is None:
                    raise ValueError("Failed to create write range")

                target_range = write_range.resize(csv_row_count, csv_col_count)
                if target_range is None:
                    raise ValueError("Failed to resize write range")

                target_range.value = data_to_write.values
            except Exception as e:
                raise ValueError(f"Error writing data: {str(e)}")

            progress_print("Saving workbook...")
            wb.save()
            progress_print("Import completed successfully.")

        except Exception as e:
            progress_print(f"An error occurred: {str(e)}")
            raise

        finally:
            # Ensure proper cleanup even if errors occur
            try:
                if wb:
                    wb.close()
                if app:
                    app.quit()
            except Exception as cleanup_error:
                progress_print(f"Warning: Error during cleanup: {str(cleanup_error)}")



    # Example of modifying an existing import function:
    def import_bbm(self, progress_print=print):
        progress_print("Importing BBM to bbm...")
        # Specify the columns to read
        columns_to_read = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

        self.import_csv_to_sheet(
            csv_file="bbm.csv",
            sheet_name="bbm",
            csv_start_row=0,
            csv_start_col=0,
            excel_start_row=2,
            excel_start_col=1,
            usecols=columns_to_read,
            progress_print=progress_print
        )
        progress_print("Done importing BBM to bbm.")

    def import_fta_entries(self, progress_print=print):
        progress_print("Importing FTA to fta...")
        self.import_csv_to_sheet(
            csv_file="fta.csv",
            sheet_name="fta",
            csv_start_row=0,
            csv_start_col=0,
            excel_start_row=2,
            excel_start_col=1,
            progress_print=progress_print
        )
        progress_print("Done importing FTA to fta.")

    def import_dk_entries(self, progress_print=print):
        progress_print("Importing DKEntries to dk_list...")
        csv_file = "entries.csv"
        excel_file = "nba.xlsm"
        sheet_name = "dk_list"

        app = None
        wb = None

        try:
            # Set up output directories
            progress_print("Setting up output directories...")
            current_folder = Path(__file__).resolve().parent
            sibling_folder = current_folder.parent / "dk_data"
            sibling_folder.mkdir(exist_ok=True)
            output_csv_file = sibling_folder / "DKSalaries.csv"

            # Read CSV data
            progress_print("Reading CSV data...")
            columns_to_read = list(range(13, 22))
            data = pd.read_csv(
                csv_file,
                skiprows=7,
                usecols=columns_to_read,
                header=0
            )

            # Verify we have data before proceeding
            if data.empty:
                raise ValueError("No data was read from the CSV file")

            # Write to output CSV
            progress_print("Writing data to output CSV...")
            data.to_csv(output_csv_file, index=False)
            progress_print(f"Data written to '{output_csv_file}'")

            # Initialize Excel
            progress_print("Opening Excel application...")
            app = xw.App(visible=False)
            app.display_alerts = False
            app.screen_updating = False

            if app is None:
                raise RuntimeError("Failed to initialize Excel application")

            progress_print("Opening Excel workbook...")
            wb = app.books.open(excel_file)

            if wb is None:
                raise RuntimeError(f"Failed to open workbook: {excel_file}")

            # Get worksheet and verify it exists
            try:
                ws = wb.sheets[sheet_name]
            except Exception as e:
                raise ValueError(f"Sheet '{sheet_name}' not found in workbook") from e

            progress_print("Clearing existing data...")
            # Get the used range before clearing
            used_range = ws.range("B2").expand()
            if used_range:  # Verify range exists before clearing
                used_range.clear_contents()

            progress_print("Writing new data...")
            # Verify data dimensions before writing
            target_range = ws.range("B2").resize(data.shape[0], data.shape[1])
            if target_range:
                target_range.value = data.values
            else:
                raise ValueError("Failed to create target range for data")

            progress_print("Saving workbook...")
            wb.save()

        except Exception as e:
            progress_print(f"Error occurred: {str(e)}")
            raise

        finally:
            # Ensure proper cleanup even if errors occur
            try:
                if wb:
                    wb.close()
                if app:
                    app.quit()
            except Exception as cleanup_error:
                progress_print(f"Warning: Error during cleanup: {str(cleanup_error)}")

        progress_print("Done importing DKEntries to dk_list.")

    def import_sog_projections(self, progress_print=print):
        progress_print("Importing SOG projections...")
        csv_file = "entries.csv"
        excel_file = "nba.xlsm"
        sheet_name = "sog_projections"

        app = None
        wb = None

        try:
            # Read CSV data first
            progress_print("Reading CSV data...")
            columns_to_read = list(range(13, 22))
            data = pd.read_csv(
                csv_file,
                skiprows=7,
                usecols=columns_to_read,
                header=0
            )

            # Verify we have data before proceeding
            if data.empty:
                raise ValueError("No data was read from the CSV file")

            progress_print("Opening Excel application...")
            app = xw.App(visible=False)

            # Check if Excel app initialized properly
            if app is None:
                raise RuntimeError("Failed to initialize Excel application")

            progress_print("Opening Excel workbook...")
            wb = app.books.open(excel_file)

            # Verify workbook opened successfully
            if wb is None:
                raise RuntimeError(f"Failed to open workbook: {excel_file}")

            # Get worksheet and verify it exists
            try:
                ws = wb.sheets[sheet_name]
            except Exception as e:
                raise ValueError(f"Sheet '{sheet_name}' not found in workbook") from e

            progress_print("Clearing existing data...")
            clear_range = ws.range(f"B2:J{360}")
            if clear_range:  # Verify range exists before clearing
                clear_range.clear_contents()

            progress_print("Writing new data...")
            # Verify data dimensions before writing
            target_range = ws.range("B2").resize(data.shape[0], data.shape[1])
            if target_range:
                target_range.value = data.values
            else:
                raise ValueError("Failed to create target range for data")

            progress_print("Saving workbook...")
            wb.save()

        except Exception as e:
            progress_print(f"Error occurred: {str(e)}")
            raise

        finally:
            # Ensure proper cleanup even if errors occur
            try:
                if wb:
                    wb.close()
                if app:
                    app.quit()
            except Exception as cleanup_error:
                progress_print(f"Warning: Error during cleanup: {str(cleanup_error)}")

        progress_print("Done importing SOG projections.")


    def import_darko(self, progress_print=print):
        progress_print("Importing DARKO to darko...")
        self.import_csv_to_sheet(
            csv_file="darko.csv",
            sheet_name="darko",
            csv_start_row=0,
            csv_start_col=0,
            excel_start_row=2,
            excel_start_col=1,
            progress_print=progress_print
        )
        progress_print("Done importing DARKO to darko.")

    def import_last10(self, progress_print = print):

        # Example API endpoint (you may need to adjust this based on your inspection)
        progress_print("Setting up API request...")

        api_url = "https://stats.nba.com/stats/leaguedashplayerstats"
        params = {
            "LastNGames": "10",
            "Season": "2024-25",
            "SeasonType": "Regular Season",
            "MeasureType": "Base",
            "PerMode": "Totals",
            "PlusMinus": "N",
            "PaceAdjust": "N",
            "Rank": "N",
            "Outcome": "",
            "Location": "",
            "Month": "0",
            "OpponentTeamID": "0",
            "VsConference": "",
            "VsDivision": "",
            "GameSegment": "",
            "Period": "0",
            "ShotClockRange": "",
            "TwoWay": "0",
            "PlayerExperience": "",
            "PlayerPosition": "",
            "StarterBench": "",
            "DraftYear": "",
            "DraftPick": "",
            "College": "",
            "Country": "",
            "Height": "",
            "Weight": "",
            "TeamID": "0",
            "Conference": "",
            "Division": "",
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
            "Referer": "https://www.nba.com/",
        }

        # Fetch data
        progress_print("Fetching data from NBA stats...")
        response = requests.get(api_url, headers=headers, params=params)
        data = response.json()

        # Extract rows and headers
        progress_print("Processing response data...")
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        csv_file_path = "last10.csv"

        progress_print("Converting data to DataFrame...")
        df = pd.DataFrame(rows, columns=headers)

        progress_print("Saving to CSV...")
        df.to_csv(csv_file_path, index=False)

        progress_print("Importing to Excel...")
        self.import_csv_to_sheet(
            csv_file=str(csv_file_path),
            sheet_name="last10",
            csv_start_row=0,
            csv_start_col=0,
            excel_start_row=2,
            excel_start_col=1,
            progress_print=progress_print
        )

        progress_print("Done importing last 10 games data.")

    # Fetch advanced team stats for the regular season
    def fetch_advanced_team_stats(self, season='2023-24', season_type='Regular Season'):
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Advanced'
        )
        return team_stats.get_data_frames()[0]


    def import_advanced(self, progress_print=print):
        progress_print("Fetching and importing Advanced Team Stats...")
        try:
            progress_print("Fetching advanced stats data...")
            season = '2024-25'
            season_type = 'Regular Season'
            df = self.fetch_advanced_team_stats(season, season_type)

            progress_print("Sorting and processing data...")
            df_sorted = df.sort_values(by='TEAM_NAME')
            csv_file_path = "advanced.csv"

            progress_print("Saving to CSV...")
            df_sorted.to_csv(csv_file_path, index=False)

            columns_to_read = [0, 1, 2, 3, 4, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 25]

            progress_print("Importing to Excel sheet...")
            self.import_csv_to_sheet(
                csv_file=str(csv_file_path),
                sheet_name="team_ratings",
                csv_start_row=0,
                csv_start_col=0,
                excel_start_row=2,
                excel_start_col=1,
                usecols=columns_to_read,
                progress_print=progress_print
            )
            progress_print("Done importing Advanced stats to team_ratings.")
        except Exception as e:
            progress_print(f"An error occurred: {e}")
            raise

    def import_traditional(self, progress_print=print):
        progress_print("Fetching and importing Traditional Team Stats...")
        try:
            progress_print("Fetching traditional stats data...")
            season = '2024-25'
            season_type = 'Regular Season'

            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star=season_type,
                measure_type_detailed_defense='Base',
                per_mode_detailed='Totals'
            )

            progress_print("Processing data...")
            df = team_stats.get_data_frames()[0]
            df_sorted = df.sort_values(by='TEAM_NAME')
            csv_file_path = "traditional.csv"

            progress_print("Saving to CSV...")
            df_sorted.to_csv(csv_file_path, index=False)

            columns_to_read = list(range(28))  # 0 through 27

            progress_print("Importing to Excel sheet...")
            self.import_csv_to_sheet(
                csv_file=str(csv_file_path),
                sheet_name="team_ratings",
                csv_start_row=0,
                csv_start_col=0,
                excel_start_row=2,
                excel_start_col=24,
                usecols=columns_to_read,
                progress_print=progress_print
            )
            progress_print("Done importing Traditional stats to team_ratings.")
        except Exception as e:
            progress_print(f"An error occurred: {e}")
            raise


    from datetime import datetime, timezone, timedelta
    def fetch_and_save_team_data_with_odds(self, progress_print=print):

        try:
            # The Odds API Key and Endpoints
            api_key = 'd4237a37fb55c03282af5de33235e1d6'
            events_url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey={api_key}&dateFormat=iso'
            odds_url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={api_key}&regions=us&markets=spreads,totals'
            progress_print("Fetching events data...")
            # Fetch events data
            response_events = requests.get(events_url)
            if response_events.status_code != 200:
                raise Exception(f"Error fetching events: {response_events.status_code} - {response_events.text}")

            events_data = response_events.json()

            progress_print("Fetching odds data...")
            # Fetch odds data
            response_odds = requests.get(odds_url)
            if response_odds.status_code != 200:
                raise Exception(f"Error fetching odds: {response_odds.status_code} - {response_odds.text}")

            odds_data = response_odds.json()

            progress_print("Processing team data...")
            # Prepare team data from events
            teams = []
            for event in events_data:
                # Parse game start time and convert to EST
                game_time_utc = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
                game_time_est = game_time_utc.astimezone(timezone(timedelta(hours=-5)))  # EST is UTC-5

                # Add each team and game time to the list
                teams.append({
                    "Team": event["home_team"],
                    "Game Time (EST)": game_time_est.strftime("%Y-%m-%d %I:%M %p"),
                    "Spread": None,
                    "Total": None
                })
                teams.append({
                    "Team": event["away_team"],
                    "Game Time (EST)": game_time_est.strftime("%Y-%m-%d %I:%M %p"),
                    "Spread": None,
                    "Total": None
                })

            progress_print("Creating DataFrame...")
            # Convert teams data to DataFrame
            teams_df = pd.DataFrame(teams)

            progress_print("Processing odds data...")
            # If the CSV already exists, read its content to preserve existing values
            csv_path = "odds.csv"
            if os.path.exists(csv_path):
                existing_data = pd.read_csv(csv_path)
                teams_df = pd.merge(teams_df, existing_data, on=["Team", "Game Time (EST)"], how="left",
                                    suffixes=("", "_existing"))

                # Preserve existing values if new data is not available
                teams_df["Spread"] = teams_df.apply(
                    lambda row: row["Spread"] if not pd.isna(row["Spread"]) else row["Spread_existing"], axis=1
                )
                teams_df["Total"] = teams_df.apply(
                    lambda row: row["Total"] if not pd.isna(row["Total"]) else row["Total_existing"], axis=1
                )

                # Drop the merged columns
                teams_df = teams_df.drop(columns=["Spread_existing", "Total_existing"])

            progress_print("Updating odds values...")
            # Match teams with odds data and update the DataFrame
            for odds_event in odds_data:
                progress_print(f"Processing odds for game: {odds_event['home_team']} vs {odds_event['away_team']}")
                bookmaker = next((b for b in odds_event["bookmakers"] if b["title"] == "DraftKings"), None)
                if not bookmaker:
                    continue

                spread_market = None
                total_market = None
                for market in bookmaker.get("markets", []):
                    if market["key"] == "spreads":
                        spread_market = market
                    elif market["key"] == "totals":
                        total_market = market

                # Update Spread values
                if spread_market:
                    for outcome in spread_market.get("outcomes", []):
                        teams_df.loc[teams_df["Team"] == outcome["name"], "Spread"] = outcome.get("point", "N/A")

                # Update Total values
                if total_market:
                    total_value = next((outcome.get("point", "N/A") for outcome in total_market.get("outcomes", [])),
                                       None)
                    if total_value is not None:
                        teams_df.loc[teams_df["Team"].isin(
                            [odds_event["home_team"], odds_event["away_team"]]), "Total"] = total_value

            progress_print("Saving to CSV...")
            # Save the updated DataFrame to a CSV file
            teams_df.to_csv(csv_path, index=False)

            # Print success messages
            print("Fetched NBA Team Data with Odds:")
            print(teams_df)
            print(f"Data successfully saved to {csv_path}")

            progress_print("Importing to Excel...")
            # Import to the sheet
            self.import_csv_to_sheet(
                csv_file=str(csv_path),
                sheet_name="odds",
                csv_start_row=0,
                csv_start_col=0,  # No need to filter columns here; `usecols` handles it
                excel_start_row=2,
                excel_start_col=1
            )
            progress_print("Done importing NBA Schedule to odds.")

        except Exception as e:
            progress_print(f"An error occurred: {e}")
            raise

    def export_projections(self, progress_print=print):
        progress_print("Exporting projections to CSV...")
        app = None
        wb = None

        try:
            # Set up paths
            current_folder = Path(__file__).resolve().parent
            dk_import_folder = current_folder.parent / "dk_import"
            dk_data_folder = current_folder.parent / "dk_data"

            progress_print("Creating output directory...")
            dk_data_folder.mkdir(exist_ok=True)

            excel_file = dk_import_folder / "nba.xlsm"
            output_csv_file = dk_data_folder / "projections.csv"
            sheet_name = "AceMind REPO"

            # Verify Excel file exists
            if not excel_file.exists():
                raise FileNotFoundError(f"Excel file not found: {excel_file}")

            # Initialize Excel
            progress_print("Opening Excel application...")
            app = xw.App(visible=False)
            app.display_alerts = False
            app.screen_updating = False

            if app is None:
                raise RuntimeError("Failed to initialize Excel application")

            progress_print("Opening Excel workbook...")
            wb = app.books.open(str(excel_file))

            if wb is None:
                raise RuntimeError(f"Failed to open workbook: {excel_file}")

            # Get worksheet and verify it exists
            try:
                ws = wb.sheets[sheet_name]
            except Exception as e:
                raise ValueError(f"Sheet '{sheet_name}' not found in workbook") from e

            progress_print("Reading sheet data...")
            # Get initial range and verify it exists
            initial_range = ws.range("A1")
            if initial_range is None:
                raise ValueError("Failed to get initial range at 'A1'")

            # Expand the range and verify expansion worked
            expanded_range = initial_range.expand()
            if expanded_range is None:
                raise ValueError("Failed to expand range from 'A1'")

            # Get the values and verify we got data
            data = expanded_range.value
            if not data:
                raise ValueError("No data found in expanded range")

            progress_print("Processing data...")
            end_index = next(
                (i for i, row in enumerate(data) if row[0] in [None, "", " "]),
                len(data)
            )
            filtered_data = data[:end_index]

            if not filtered_data:
                raise ValueError("No valid data found after filtering")

            progress_print("Writing to CSV...")
            with open(output_csv_file, "w", newline="") as file:
                rows_processed = 0
                total_rows = len(filtered_data)
                for row in filtered_data:
                    if all(cell in [None, "", " "] for cell in row) or row[0] == 0:
                        continue
                    try:
                        formatted_row = [
                            int(cell) if col_idx == 4 and isinstance(cell, float) and cell.is_integer() else
                            f"{cell:.2f}" if isinstance(cell, float) and col_idx != 4 else
                            cell if cell not in [None, "", " "] else ""
                            for col_idx, cell in enumerate(row)
                        ]
                        file.write(",".join(str(cell) for cell in formatted_row) + "\n")
                        rows_processed += 1
                        if rows_processed % 100 == 0:
                            progress_print(f"Processed {rows_processed}/{total_rows} rows...")
                    except Exception as row_error:
                        progress_print(f"Warning: Error processing row {rows_processed + 1}: {str(row_error)}")
                        continue

            progress_print(f"Projections successfully exported to '{output_csv_file}'.")

        except Exception as e:
            progress_print(f"An error occurred: {str(e)}")
            raise

        finally:
            # Ensure proper cleanup even if errors occur
            try:
                if wb:
                    wb.close()
                if app:
                    app.quit()
            except Exception as cleanup_error:
                progress_print(f"Warning: Error during cleanup: {str(cleanup_error)}")

    def run_all_imports(self, progress_print=print):
        progress_print("Running all imports...")
        self.import_bbm(progress_print=progress_print)
        self.import_fta_entries(progress_print=progress_print)
        self.import_dk_entries(progress_print=progress_print)
        self.import_sog_projections(progress_print=progress_print)
        self.import_darko(progress_print=progress_print)
        self.import_advanced(progress_print=progress_print)
        self.import_traditional(progress_print=progress_print)
        progress_print("All imports completed.")

    # Post-process predictions to apply the 1% ownership cap for low minutes players
    def apply_low_minutes_cap(self, predictions, df):
        capped_predictions = predictions.copy()
        capped_predictions[df['Low_Minutes_Flag'] == 1] = np.minimum(
            capped_predictions[df['Low_Minutes_Flag'] == 1], 0.01)
        return capped_predictions



    def ownership_projections(self, progress_print=print):
        import os
        import pandas as pd
        import pickle
        import xlwings as xw
        import random
        from scipy.optimize import minimize

        def generate_random_scores(players, num_samples, sd_multiplier=1, progress_print=print):
            progress_print("Generating random scores...")
            random_scores = {}
            total_players = len(players)
            for idx, player in enumerate(players, 1):
                if idx % 20 == 0:  # Update every 20 players
                    progress_print(f"Processing player {idx}/{total_players}")

                proj_points = player['Points Proj']
                ceiling = player['Ceiling']
                std_dev = ceiling / sd_multiplier
                scores = np.random.normal(loc=proj_points, scale=std_dev, size=num_samples)
                random_scores[player['DK Name']] = scores

            progress_print("Random scores generation complete")
            return random_scores

        def simulate_weighted_feasibility_with_progress(data, max_salary=50000, lineup_size=8,
                                                        num_samples=50000, print_every=1000,
                                                        progress_print=print):
            slot_map = {
                1: ['PG', 'PG/SG', 'PG/SF', 'G'],
                2: ['SG', 'PG/SG', 'G'],
                3: ['SF', 'SG/SF', 'PG/SF', 'SF/PF', 'F'],
                4: ['PF', 'SF/PF', 'PF/C', 'F'],
                5: ['C', 'PF/C'],
                6: ['PG', 'SG', 'PG/SG', 'SG/SF', 'PG/SF', 'G'],
                7: ['SF', 'PF', 'SF/PF', 'SG/SF', 'PG/SF', 'F'],
                8: ['PG', 'SG', 'SF', 'PF', 'C', 'PG/SG', 'SG/SF', 'SF/PF', 'PG/SF', 'PF/C', 'G', 'F', 'UTIL'],
            }

            progress_print("Setting up simulation parameters...")
            min_salary = 49000
            min_score = 200
            min_proj_points = 12

            eligible_players = data[data['Points Proj'] >= min_proj_points].to_dict(orient='records')
            weighted_feasibility = {player['DK Name']: 0 for player in eligible_players}
            tournament_feasibility = {player['DK Name']: 0 for player in eligible_players}
            count = len(eligible_players)

            progress_print(f"Starting simulation with {count} eligible players...")
            lineups = []

            progress_print("Generating random scores...")
            random_scores = generate_random_scores(eligible_players, num_samples, progress_print=progress_print)

            progress_print("Beginning lineup generation...")
            i = 1
            while i <= num_samples:
                if i % print_every == 0:
                    progress_print(f"Processing lineup {i}/{num_samples}")

                lineup = []
                selected_players = set()

                for slot in range(1, lineup_size + 1):
                    slot_positions = slot_map[slot]
                    slot_eligible_players = [
                        player for player in eligible_players
                        if player['DK Name'] not in selected_players and player['Position'] in slot_positions
                    ]

                    if not slot_eligible_players:
                        break

                    selected_player = random.choice(slot_eligible_players)
                    lineup.append(selected_player)
                    selected_players.add(selected_player['DK Name'])

                if len(lineup) == lineup_size:
                    total_salary = sum(player['Salary'] for player in lineup)
                    total_points = sum(random.choice(random_scores[player['DK Name']]) for player in lineup)

                    if min_salary < total_salary <= max_salary and total_points > min_score:
                        i += 1
                        for player in lineup:
                            weighted_feasibility[player['DK Name']] += 1
                        lineups.append({
                            'lineup': lineup,
                            'total_points': total_points
                        })

            progress_print("Sorting lineups by total points...")
            lineups.sort(key=lambda x: x['total_points'], reverse=True)

            progress_print("Processing top lineups...")
            top_lineups = int(num_samples * 0.20)
            lineups = lineups[:top_lineups]

            progress_print("Calculating tournament feasibility scores...")
            total_top_lineups = len(lineups)
            for idx, (rank, lineup_info) in enumerate(enumerate(lineups, start=1), 1):
                if idx % 1000 == 0:
                    progress_print(f"Processing top lineup {idx}/{total_top_lineups}")
                lineup = lineup_info['lineup']
                for player in lineup:
                    tournament_feasibility[player['DK Name']] += 1 / rank

            progress_print("Tournament feasibility calculation completed.")
            return weighted_feasibility, tournament_feasibility

        # Similar progress updates for simulate_feasibility_with_progress...
        def simulate_feasibility_with_progress(data, max_salary=50000, lineup_size=8, num_samples=1000000,
                                               print_every=100000, progress_print=print):
            # Simulate valid lineups with random sampling and progress updates.
            min_salary = 0
            min_score = 0
            min_proj_points = 0
            # players = data[['DK Name', 'Salary', 'Points Proj', 'Ceiling']].to_dict(orient='records')
            eligible_players = data[data['Points Proj'] >= min_proj_points][
                ['DK Name', 'Salary', 'Points Proj', 'Ceiling']].to_dict(orient='records')
            feasibility = {player['DK Name']: 0 for player in eligible_players}  # Initialize feasibility counts

            print(f"Simulating {num_samples} random lineups...")

            i = 1  # Initialize i outside the loop
            while i <= num_samples:  # Use while instead of for to control incrementing i manually
                lineup = random.sample(eligible_players, lineup_size)  # Randomly select 8 players
                total_salary = sum(player['Salary'] for player in lineup)
                total_points = sum(player['Points Proj'] for player in lineup)

                # Only process lineups with total_salary > 49000
                if min_salary < total_salary <= max_salary and total_points > min_score:  # Check salary range
                    for player in lineup:
                        feasibility[player['DK Name']] += 1
                    i += 1  # Increment i only if the lineup meets the salary criteria

                # Print progress every N samples
                if i % print_every == 0:
                    print(f"Processed {i} / {num_samples} lineups...")

            # Normalize feasibility scores
            for player in feasibility:
                feasibility[player] /= num_samples

            print("Feasibility calculation completed.")
            return feasibility

        try:
            progress_print("Starting ownership projections process...")

            progress_print("Loading trained model...")
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, '..', 'src', 'final_nba_model.pkl')
            with open(model_path, 'rb') as file:
                model = pickle.load(file)

            progress_print("Opening Excel workbook...")
            file_path = os.path.join('..', 'dk_import', 'nba.xlsm')
            app = xw.App(visible=False)
            wb = app.books.open(file_path)
            sheet = wb.sheets["ownership"]

            progress_print("Reading and processing data...")
            # Get initial range and verify it exists
            initial_range = sheet.range('A1')
            if initial_range is None:
                raise ValueError("Failed to get initial range at 'A1'")

            # Expand the range and verify expansion worked
            expanded_range = initial_range.expand()
            if expanded_range is None:
                raise ValueError("Failed to expand range from 'A1'")

            # Get the values and verify we got data
            data = expanded_range.value
            if not data:
                raise ValueError("No data found in expanded range")
            #data = sheet.range('A1').expand().value
            columns = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=columns)

            # Drop rows where 'DK Name' is empty or invalid
            df = df[df['DK Name'].notna() & (df['DK Name'] != "0") & (df['DK Name'].str.strip() != "")]

            # Ensure 'Salary' column is numeric and drop rows with invalid salaries
            df['Salary'] = pd.to_numeric(df['Salary'],
                                         errors='coerce')  # Convert to numeric, setting invalid values to NaN
            df = df.dropna(subset=['Salary'])  # Drop rows where 'Salary' is NaN
            df['Salary'] = df['Salary'].astype(int)  # Convert to integers

            df['Contest ID'] = 1
            df['Player_Pool_Size'] = df.groupby('Contest ID')['DK Name'].transform('count')
            df['Games Played'] = df['Team'].nunique() // 2

            progress_print("Cleaning data...")
            # ... your data cleaning code ...

            progress_print("Calculating weighted feasibility scores...")

            progress_print("Calculating feasibility scores...")
            feasibility = simulate_feasibility_with_progress(
                df, max_salary=1000000, lineup_size=8, num_samples=100000,
                print_every=1000, progress_print=progress_print)

            weighted_feasibility, tournament_feasibility = simulate_weighted_feasibility_with_progress(
                df, max_salary=50000, lineup_size=8, num_samples=2000,
                print_every=1000, progress_print=progress_print)


            progress_print("Processing feature engineering...")
            # ... rest of your feature engineering code ...

            df['Feasibility'] = df['DK Name'].map(feasibility) * df['Minutes']
            df['Weighted Feasibility'] = df['DK Name'].map(weighted_feasibility)
            df['Tournament Feasibility'] = df['DK Name'].map(tournament_feasibility)
            df['Proj_Salary_Ratio'] = df['Points Proj'] / df['Salary']

            # new
            df['Normalized_Value'] = df.groupby('Contest ID')['Value'].transform(lambda x: (x - x.mean()) / x.std())
            df['Prev_Salary_Mean'] = df.groupby('DK Name')['Salary'].shift(1).rolling(window=3).mean()
            df['Log_Proj'] = np.log1p(df['Points Proj'])

            # Add interaction features
            df['Salary_Proj_Interaction'] = df['Salary'] * df['Points Proj']
            df['Proj_Feasibility_Interaction'] = df['Points Proj'] * df['Feasibility']
            df['Value_Feasibility_Interaction'] = df['Value'] * df['Feasibility']

            # df['Ceiling'] = (df['Points Proj'] + df['Ceiling'])
            df['Low_Minutes_Flag'] = (df['Minutes'] <= 12).astype(int)
            df['Low_Minutes_Penalty'] = df['Low_Minutes_Flag'] * df['Points Proj']

            df['Team_Total_Points_Proj'] = df.groupby('Team')['Points Proj'].transform('sum')
            df['Player_Points_Percentage'] = df['Points Proj'] / df['Team_Total_Points_Proj']

            df['Value_Plus'] = df['Points Proj'] - df['Salary'] / 1000 * 4
            df['Ceiling_Plus'] = (df['Points Proj'] + df['Ceiling']) - df['Salary'] / 1000 * 5

            # Step 5: Define the exact feature list used during training
            features = [
                'Minutes',
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

            # Function to compute dynamic shift
            def compute_dynamic_shift(rank, total, shift_start, shift_end):
                scale = (rank / total) ** drift
                return shift_start + (shift_end - shift_start) * scale

            def rescale_with_bounds(predictions: pd.Series, target_sd=7.0, min_bound=0.0, top_boost=1.15):
                numbers = predictions.values
                index = predictions.index
                original_mean = np.mean(numbers)

                # Split into small (<1) and large values
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

                    # Allow mean to go higher but not lower
                    mean_penalty = (np.maximum(0, original_mean - current_mean)) ** 2 * 800

                    # Penalties
                    sd_penalty = (current_sd - target_sd) ** 2 * 3000
                    min_penalty = np.sum(np.maximum(0, min_bound - result) ** 2) * 1000

                    # Penalty for small values order violation
                    order_penalty = 0
                    if np.any(small_mask):
                        small_diffs = np.diff(result[small_mask])
                        order_penalty = np.sum(np.maximum(0, -small_diffs)) * 2000

                    return sd_penalty + mean_penalty + min_penalty + order_penalty

                # Initial guess with higher scaling factor
                initial_guess = [
                    (target_sd / np.std(numbers)) * 1.1,  # increased initial scale for large values
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

                # Boost top values
                if np.any(large_mask):
                    # Find top 10% threshold
                    top_threshold = np.percentile(final_result, 90)
                    top_mask = final_result >= top_threshold

                    # Apply progressive boost (more boost for higher values)
                    if np.any(top_mask):
                        boost_range = final_result[top_mask] - top_threshold
                        max_boost = boost_range / (np.max(final_result) - top_threshold)
                        boost_factor = 1 + (max_boost * (top_boost - 1))
                        final_result[top_mask] *= boost_factor

                # Ensure minimum bound
                final_result = np.maximum(final_result, min_bound)

                return pd.Series(final_result, index=index)

            def predict_sd(players):
                a = 0.0001  # Coefficient for Players^2
                b = -0.0770  # Coefficient for Players
                c = 21.0  # Intercept
                players = np.array(players)  # Ensure the input is a NumPy array
                return a * (players ** 2) + b * players + c

            # Set the global option to display all columns
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            from tabulate import tabulate
            print("Generating predictions...")

            shift_start = .35
            shift_end = 1.00
            add = 1.0  # Constant to avoid log issues with zero
            drift = 1.35
            players_per_game = 48  # Number of players per game to rank for the shift
            alpha = 0.99  # Weight for `Tournament Feasibility` in `Custom_Target`

            # Calculate Custom_Target
            df['Custom_Target'] = alpha * df['Tournament Feasibility'] + (1 - alpha) * df['Own Actual']

            # Rank players by preliminary predictions
            prediction_df = df[features]
            preliminary_predictions = model.predict(prediction_df)
            df['Preliminary_Predictions'] = preliminary_predictions
            rank_limit = df['Games Played'].iloc[0] * players_per_game

            # Rank players and calculate dynamic shift
            df['Group_Rank'] = df['Custom_Target'].rank(ascending=False)
            # df['Group_Rank'] = df['Preliminary_Predictions'].rank(ascending=False)
            df['Group_Size'] = len(df)  # Total players in the contest
            df['Dynamic_Shift'] = df.apply(
                lambda row: compute_dynamic_shift(
                    row['Group_Rank'], rank_limit, shift_start, shift_end
                ) if row['Group_Rank'] <= rank_limit else shift_start,
                axis=1
            )

            # Apply the dynamic shift to predictions
            predictions = (preliminary_predictions ** (1 / df['Dynamic_Shift'])) - add

            # Scale predictions to sum to 800
            scaling_factor = 800 / predictions.sum()
            predictions = predictions * scaling_factor

            # Apply post-processing adjustments
            df['Predicted Ownership'] = self.apply_low_minutes_cap(predictions, df)
            threshold = 1e-6  # Values smaller than this will be treated as 0
            df.loc[df['Points Proj'].abs() < threshold, 'Predicted Ownership'] = (
                    df.loc[df['Points Proj'].abs() < threshold, 'Salary'] / 100000
            )
            df['Predicted Ownership'] = df['Predicted Ownership'].clip(lower=0)  # Set the minimum threshold
            # df['Predicted Ownership'] = predictions

            predictions = df['Predicted Ownership']
            current_sum = predictions.sum()

            player_pool = len(df)
            estimated_sd = predict_sd(player_pool)
            estimated_sd_multiplier = 1.00 * (current_sum / 800.0)  # Set your desired SD value
            target_sd = estimated_sd * estimated_sd_multiplier

            print(f'Players : {player_pool} Sum: {current_sum} ')
            print(f'Estimated SD: {estimated_sd} Multiplier: {estimated_sd_multiplier} ')
            print(f'Target SD: {target_sd}')

            # adjusted_predictions = adjust_sd(predictions, target_sd, target_sum)
            adjusted_predictions = rescale_with_bounds(predictions, target_sd=target_sd, top_boost=1.50)

            final_predictions = self.apply_low_minutes_cap(adjusted_predictions, df)
            df['Predicted Ownership'] = final_predictions
            threshold = 1e-6  # Values smaller than this will be treated as 0
            df.loc[df['Points Proj'].abs() < threshold, 'Predicted Ownership'] = (
                    df.loc[df['Points Proj'].abs() < threshold, 'Salary'] / 100000
            )
            df['Predicted Ownership'] = df['Predicted Ownership'].clip(lower=0)  # Set the minimum threshold

            final_predictions = df['Predicted Ownership']
            scaling_factor = 800 / final_predictions.sum()
            final_predictions *= scaling_factor
            df['Predicted Ownership'] = final_predictions

            # Debugging: Print before scaling and final results
            print(
                f"Before Scaling - Mean: {predictions.mean():.4f}, SD: {predictions.std():.4f} Total: {predictions.sum()}")
            print(
                f"After Scaling Adjusted - Mean: {adjusted_predictions.mean():.4f}, SD: {adjusted_predictions.std():.4f}, Total: {adjusted_predictions.sum():.4f}")
            print(
                f"After Scaling Final - Mean: {final_predictions.mean():.4f}, SD: {final_predictions.std():.4f}, Total: {final_predictions.sum():.4f}")

            print(f"\nPredicted Ownership Total: {df['Predicted Ownership'].sum():.2f}")
            print(
                df[['DK Name', 'Predicted Ownership']].sort_values(by='Predicted Ownership', ascending=False).head(25))

            # Define the keys to display
            keys_to_display = ['DK Name', 'Points Proj', 'Predicted Ownership', 'Dynamic_Shift']
            result_df = df[keys_to_display].copy()
            result_df.loc[:, 'Predicted Ownership'] = result_df['Predicted Ownership'].round(2)
            result_df.loc[:, 'Dynamic_Shift'] = result_df['Dynamic_Shift'].round(3)
            formatted_table = tabulate(result_df.head(50), headers='keys', tablefmt='pretty', showindex=False)
            progress_print(formatted_table)

            progress_print("Writing results back to Excel...")
            predicted_ownership_col_index = columns.index('Own Proj') + 1
            predicted_values = df['Predicted Ownership'].values.tolist()
            sheet.range(2, predicted_ownership_col_index).value = [[v] for v in predicted_values]

            progress_print("Saving workbook...")
            wb.save()
            progress_print("Ownership projections completed successfully!")

        except Exception as e:
            progress_print(f"An error occurred: {e}")
            raise
        finally:
            wb.close()
            app.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImportTool()
    window.show()
    sys.exit(app.exec())
