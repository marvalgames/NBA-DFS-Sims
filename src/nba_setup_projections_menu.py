import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os


import numpy as np
import requests

# In your MainApp class, add these imports at the top:
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from nba_api.stats.endpoints import leaguedashteamstats


from daily_download import DailyDownload
from nba_data_manager import NBADataManager
from nba_minutes_prediction_setup import NbaMInutesPredictions
from nba_minutes_predictions_enhanced import PredictMinutes



from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget, QLabel, QMessageBox, QTextEdit, QProgressDialog, QHBoxLayout,
                             QTableView, QSplitter, QComboBox, QGroupBox)
from PyQt6.QtCore import QAbstractTableModel, Qt
import pandas as pd

from PyQt6.QtCore import QSortFilterProxyModel

class CustomSortFilterProxyModel(QSortFilterProxyModel):
    def lessThan(self, left, right):
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.EditRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.EditRole)

        try:
            # Try numeric comparison first
            return float(left_data) < float(right_data)
        except (ValueError, TypeError):
            # Fall back to string comparison if numeric fails
            return str(left_data) < str(right_data)

class DataFrameModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data
        self._original_data = data.copy()

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole or Qt.ItemDataRole.EditRole):
        if not index.isValid():
            return None

        value = self._data.iloc[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.float64)):
                return f"{value:.2f}"
            return str(value)

        elif role == Qt.ItemDataRole.EditRole:
            # Return the actual value for sorting
            return float(value) if isinstance(value, (int, float, np.number)) else str(value)

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float, np.number)):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return None

class ImportThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, object)  # Added object for DataFrame

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            original_print = print

            def progress_print(*args):
                message = ' '.join(map(str, args))
                self.progress.emit(message)
                original_print(*args)

            self.kwargs['progress_print'] = progress_print
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(True, "Import completed successfully!", result)
        except Exception as e:
            self.finished.emit(False, str(e), None)


class ImportTool(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the file paths
        self.setup_file_paths()

        self.setWindowTitle("NBA DFS Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize data storage
        self.dataframes = {}
        """Set up the main UI with two sections: data viewer and import controls"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create a splitter for left and right sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side - Control Panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        # Data Viewer Controls
        viewer_group = QGroupBox("Data Viewer")
        viewer_layout = QVBoxLayout()
        self.data_selector = QComboBox()
        self.data_selector.addItems(['BBM', 'FTA', 'DK Entries', 'Darko', 'Team Stats', 'Odds'])
        self.data_selector.currentTextChanged.connect(self.on_data_selection_changed)
        viewer_layout.addWidget(self.data_selector)
        viewer_group.setLayout(viewer_layout)
        left_layout.addWidget(viewer_group)

        # Import Controls
        import_group = QGroupBox("Import Controls")
        import_layout = QVBoxLayout()

        # Daily Operations
        daily_button = QPushButton("Download Daily Data")
        daily_button.clicked.connect(lambda: self.run_threaded_import(self.import_daily))
        import_layout.addWidget(daily_button)

        game_logs_button = QPushButton("Expand Daily Game Logs")
        game_logs_button.clicked.connect(lambda: self.run_threaded_import(self.expand_game_logs))
        import_layout.addWidget(game_logs_button)

        # Import Buttons
        bbm_button = QPushButton("Import BBM")
        bbm_button.clicked.connect(lambda: self.run_threaded_import(self.import_bbm))
        import_layout.addWidget(bbm_button)

        fta_button = QPushButton("Import FTA")
        fta_button.clicked.connect(lambda: self.run_threaded_import(self.import_fta_entries))
        import_layout.addWidget(fta_button)

        dk_button = QPushButton("Import DK Entries")
        dk_button.clicked.connect(lambda: self.run_threaded_import(self.import_dk_entries))
        import_layout.addWidget(dk_button)

        darko_button = QPushButton("Import Darko")
        darko_button.clicked.connect(lambda: self.run_threaded_import(self.import_darko))
        import_layout.addWidget(darko_button)

        team_stats_button = QPushButton("Import Team Stats")
        team_stats_button.clicked.connect(lambda: self.run_threaded_import(self.import_team_stats))
        import_layout.addWidget(team_stats_button)

        # Additional Operations
        odds_button = QPushButton("Import NBA Game Odds")
        odds_button.clicked.connect(lambda: self.run_threaded_import(self.fetch_and_save_team_data_with_odds))
        import_layout.addWidget(odds_button)

        minutes_button = QPushButton("Predict Player Minutes")
        minutes_button.clicked.connect(lambda: self.run_threaded_import(self.predict_minutes))
        import_layout.addWidget(minutes_button)

        own_button = QPushButton("Calculate Ownership")
        own_button.clicked.connect(lambda: self.run_threaded_import(self.ownership_projections))
        import_layout.addWidget(own_button)

        export_button = QPushButton("Export Projections")
        export_button.clicked.connect(lambda: self.run_threaded_import(self.export_projections))
        import_layout.addWidget(export_button)

        all_button = QPushButton("Run All Imports")
        all_button.clicked.connect(lambda: self.run_threaded_import(self.run_all_imports))
        import_layout.addWidget(all_button)

        # Add quit button
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.close)
        import_layout.addWidget(quit_button)

        import_group.setLayout(import_layout)
        left_layout.addWidget(import_group)

        # Right side - Data View and Progress
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        splitter.addWidget(right_widget)

        # Add table view
        self.setup_table_view()
        right_layout.addWidget(self.table_view)

        # Add progress display
        self.progress_display = QTextEdit()
        self.progress_display.setReadOnly(True)
        self.progress_display.setMaximumHeight(100)
        right_layout.addWidget(self.progress_display)

        # Set splitter sizes
        splitter.setSizes([300, 900])



    def run_threaded_import(self, func):
        try:
            self.import_thread = ImportThread(func)
            self.import_thread.progress.connect(self.update_progress)
            self.import_thread.finished.connect(self.import_finished)
            self.import_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def update_progress(self, message):
        self.progress_display.append(message)
        QApplication.processEvents()

    def import_finished(self, success, message, data):
        if success:
            if data is not None and isinstance(data, pd.DataFrame):
                current_selection = self.data_selector.currentText()
                self.dataframes[current_selection] = data
                self.display_dataframe(data)
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", f"An error occurred: {message}")

    def setup_file_paths(self):
        """Setup the file paths for data files"""
        # Get the current script's directory
        current_folder = Path(__file__).parent
        # Get the sibling dk_import folder
        self.import_folder = current_folder.parent / "dk_import"
        # Get the sibling dk_data folder
        self.data_folder = current_folder.parent / "dk_data"

        # Verify folders exist
        if not self.import_folder.exists():
            raise FileNotFoundError(f"Import folder not found: {self.import_folder}")

        # Create data folder if it doesn't exist
        self.data_folder.mkdir(exist_ok=True)

        print(f"Import folder: {self.import_folder}")
        print(f"Data folder: {self.data_folder}")



    def import_daily(self, progress_print=print):
        downloader = DailyDownload()
        downloader.copy_dk_entries()
        downloader.download_all()
        USERNAME = "marvalgames"
        PASSWORD = "NWMUCBPOUD"
        downloader.download_and_rename_csv(USERNAME, PASSWORD)

        print('Completed')

        progress_print("Done downloading data.")

    def standardize_player_names(self, df, name_column):
        """
        Standardizes player names according to specific rules
        """
        if not isinstance(df, pd.DataFrame):
            return df

        # Create a copy to avoid modifying the original
        df = df.copy()

        name_mappings = {
            "Herb Jones": "Herbert Jones",
            "GG Jackson": "Gregory Jackson",
            "G.G. Jackson": "Gregory Jackson",
            "Alexandre Sarr": "Alex Sarr",
            "Yongxi Cui": "Cui Yongxi",
            "Nicolas Claxton": "Nic Claxton",
            "Cameron Johnson": "Cam Johnson",
            "Kenyon Martin Jr": "KJ Martin",
            "Ronald Holland": "Ron Holland",
            "Nah'Shon Hyland": "Bones Hyland",
            "Elijah Harkless": "EJ Harkless",
            "Cameron Payne": "Cam Payne",
            "Bub Carrington": "Carlton Carrington",
            "Jabari Smith Jr": "Jabari Smith",
            "Gary Trent Jr": "Gary Trent",
            "Tim Hardaway Jr": "Tim Hardaway",
            "Michael Porter Jr": "Michael Porter",
            "Kelly Oubre Jr": "Kelly Oubre",
            "Patrick Baldwin Jr": "Patrick Baldwin",
            "Kevin Knox II": "Kevin Knox",
            "Trey Murphy III": "Trey Murphy",
            "Wendell Moore Jr": "Wendell Moore",
            "Vernon Carey Jr": "Vernon Carey",
            # ... your existing mappings ...
            "Kristaps PorziÅ†Ä£is": "Kristaps Porzingis"  # Add this specific mapping if needed
        }

        def remove_accents(text):
            if not isinstance(text, str):
                return text
            try:
                # Normalize to decomposed form (separate letters from accents)
                nfkd_form = unicodedata.normalize('NFKD', text)
                # Remove non-ASCII characters (like accents)
                return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
            except:
                # Fallback method if Unicode normalization fails
                accents = "áàâãäéèêëíìîïóòôõöúùûüýÿÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÝćčćdšžCCÐŠŽūÅ†Ä£"
                regular = "aaaaaeeeeiiiiooooouuuuyyAAAAAEEEEIIIIOOOOOUUUUYcccdszCCDSZuAng"

                for accent, reg in zip(accents, regular):
                    text = text.replace(accent, reg)
                return text

        # Apply standardization to the name column
        df[name_column] = df[name_column].apply(lambda x: x if not isinstance(x, str) else x.strip())

        # Apply name mappings
        for old_name, new_name in name_mappings.items():
            df[name_column] = df[name_column].apply(
                lambda x: new_name if isinstance(x, str) and x.strip() == old_name else x
            )

        # Remove accents and special characters
        df[name_column] = df[name_column].apply(
            lambda x: remove_accents(x) if isinstance(x, str) else x
        )

        # Remove suffixes and special characters
        df[name_column] = df[name_column].apply(
            lambda x: x.replace(" Jr", "").replace("III", "").replace("II", "").replace("'", "").replace(".",
                                                                                                         "").strip()
            if isinstance(x, str) else x
        )

        return df

    def merge_latest_records_with_columns(self, excel_df, combined_df,
                                          excel_key='Player',
                                          combined_key='PLAYER_NAME',
                                          date_column='Game_Date',
                                          columns_to_merge=None):
        """
        Merges DataFrames with different key column names and writes to both Excel and CSV
        """
        print("Starting merge process...")
        print(f"Excel DataFrame contains {len(excel_df)} rows")
        print(f"Combined DataFrame contains {len(combined_df)} rows")

        # First, identify rows with actual players
        excel_df['has_player'] = excel_df[excel_key].notna() & (excel_df[excel_key] != '')

        print("\nStandardizing player names...")
        excel_df = self.standardize_player_names(excel_df, excel_key)
        combined_df = self.standardize_player_names(combined_df, combined_key)

        print("Creating working copy of data...")
        working_df = combined_df.copy()

        # Convert date column to datetime
        working_df[date_column] = pd.to_datetime(working_df[date_column])

        # Rename the key column to match excel_df
        working_df = working_df.rename(columns={combined_key: excel_key})

        # If no columns specified, use all columns from working_df
        if columns_to_merge is None:
            columns_to_merge = working_df.columns.tolist()

        # Ensure key and date columns are included
        required_columns = [excel_key, date_column]
        columns_to_use = list(set(required_columns + columns_to_merge))

        # Filter and get latest records
        print("\nFiltering for matching players...")
        filtered_df = working_df[working_df[excel_key].isin(excel_df[excel_key])]
        print(f"Found {len(filtered_df)} matching records")

        print("Getting latest records for each player...")
        latest_records = (filtered_df[columns_to_use]
                          .sort_values(date_column)
                          .groupby(excel_key)
                          .last()
                          .reset_index())

        print(f"Retrieved {len(latest_records)} latest records")

        # Drop date column if it wasn't in columns_to_merge
        if date_column not in columns_to_merge:
            latest_records = latest_records.drop(columns=[date_column])

        # Merge with excel_df
        result_df = excel_df.merge(latest_records,
                                   on=excel_key,
                                   how='left')

        # Handle missing values for player rows only
        print("Filling missing values with zeros for player rows only...")
        # Get numeric columns only
        numeric_columns = result_df.select_dtypes(include=['int64', 'float64']).columns
        columns_to_fill = [col for col in numeric_columns
                           if col != excel_key and col != 'has_player']

        # Create a mask for player rows
        player_mask = result_df['has_player']

        # Fill values for numeric columns only
        for col in columns_to_fill:
            null_mask = result_df[col].isna()
            if null_mask.any():
                result_df.loc[player_mask & null_mask, col] = 0

        # Drop the helper column
        result_df = result_df.drop(columns=['has_player'])

        print(f"\nMerge complete! Final DataFrame contains {len(result_df)} rows")

        # Print summary of matches/non-matches
        matched_players = result_df[result_df[columns_to_fill[0]] != 0].shape[0]
        print(f"Players with matching data: {matched_players}")
        print(f"Players without matching data: {len(result_df) - matched_players}")

        return result_df

    def process_game_logs(self):
        # Read from Excel using XWings

        time.sleep(1)  # Give Excel a moment to fully initialize
        # Connect to Excel
        # excel_path = os.path.join('..', 'dk_import', 'nba - Copy.xlsm')
        csv_path = os.path.join('..', 'dk_import', 'nba_daily_combined.csv')
        csv_read = os.path.join('..', 'dk_import', 'nba_boxscores_enhanced.csv')


        # Read data
        # needed_data = {
        #     'Player': ws.range(f'A2:A{last_row}').value,
        #     'Team': ws.range(f'B2:B{last_row}').value,
        #     'Position': ws.range(f'C2:C{last_row}').value,
        #     'Salary': ws.range(f'D2:D{last_row}').value,
        #     'Minutes': ws.range(f'E2:E{last_row}').value,
        #     'Max Minutes': ws.range(f'J2:J{last_row}').value,
        #     'Projection': ws.range(f'M2:M{last_row}').value,
        #
        # }

        needed_data = {}

        df = pd.DataFrame(needed_data)

        # Load your combined DataFrame
        combined_df = pd.read_csv(csv_read, encoding='utf-8')  # or however you load it
        result_df = self.merge_latest_records_with_columns(df, combined_df, excel_key='Player',
                                                      combined_key='PLAYER_NAME', date_column="GAME_DATE")
        result_df = result_df.rename(columns={'Projection': 'Projected Pts'})
        # Define the new column order
        new_column_order = [
            # Core Player Info
            'Player', 'Team', 'Position', 'Salary',

            # Minutes Data
            'Minutes', 'Max Minutes', 'MIN_CUM_AVG', 'MIN_LAST_3_AVG', 'MIN_LAST_5_AVG', 'MIN_LAST_10_AVG',
            'MIN_TREND', 'MIN_CONSISTENCY', 'MIN_CONSISTENCY_SCORE',
            'MIN_ABOVE_20', 'MIN_ABOVE_25', 'MIN_ABOVE_30',
            'MIN_ABOVE_AVG_STREAK',

            'FREQ_ABOVE_20',
            'FREQ_ABOVE_25',
            'FREQ_ABOVE_30',

            # DraftKings Scoring
            'Projected Pts', 'DK', 'DK_CUM_AVG', 'DK_LAST_3_AVG', 'DK_LAST_5_AVG', 'DK_LAST_10_AVG',
            'DK_TREND', 'DK_TREND_5', 'DK_CONSISTENCY',

            # Key Stats - Recent Averages
            'PTS_LAST_3_AVG', 'PTS_LAST_5_AVG', 'PTS_LAST_10_AVG',
            'REB_LAST_3_AVG', 'REB_LAST_5_AVG', 'REB_LAST_10_AVG',
            'AST_LAST_3_AVG', 'AST_LAST_5_AVG', 'AST_LAST_10_AVG',

            # Cumulative Averages
            'PTS_CUM_AVG', 'REB_CUM_AVG', 'AST_CUM_AVG',

            # Efficiency Metrics
            'PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN',
            'SCORING_EFFICIENCY', 'RECENT_SCORING_EFF',

            # Shooting Stats
            'FG_PCT', 'FG3_PCT', 'FT_PCT',
            'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',

            # Game Impact Stats
            'PLUS_MINUS', 'PLUS_MINUS_PER_MIN',
            'PLUS MINUS_LAST_3_AVG', 'PLUS MINUS_LAST_5_AVG', 'PLUS MINUS_LAST_10_AVG',
            'PLUS MINUS_CUM_AVG', 'PLUS MINUS_TREND', 'PLUS MINUS_CONSISTENCY',

            # Trend Analysis
            'PTS_TREND', 'REB_TREND', 'AST_TREND',
            'PTS_CONSISTENCY', 'REB_CONSISTENCY', 'AST_CONSISTENCY',

            # Team Context
            'TEAM_MIN_PERCENTAGE', 'TEAM_PROJ_RANK',
            'PTS_VS_TEAM_AVG', 'REB_VS_TEAM_AVG', 'AST_VS_TEAM_AVG', 'MIN_VS_TEAM_AVG',

            # Role Analysis
            'ROLE_CHANGE_3_10', 'ROLE_CHANGE_5_10',
            'IS_TOP_3_PROJ', 'LOW_MIN_TOP_PLAYER',

            # Game Info
            'GAME_DATE', 'IS_HOME', 'IS_B2B', 'DAYS_REST',
            'MATCHUP', 'WL', 'BLOWOUT_GAME',

            # Additional Stats
            'STL', 'BLK', 'TOV', 'OREB', 'DREB', 'PF',

            # Metadata
            'PLAYER_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'SEASON_ID',
            'VIDEO_AVAILABLE'

        ]

        # Reorder the DataFrame
        result_df = result_df[new_column_order]

        result_df.to_csv(csv_path, index=False)
        return result_df


    def merge_three_dataframes(self):
        """
        Merges SOG, BBM, and game logs DataFrames based on standardized player names.
        """
        # Step 1: Load the DataFrames
        sog_df = self.import_sog_projections()
        bbm_df = self.import_bbm()
        game_logs_df = self.process_game_logs()

        # Step 2: Select Relevant Columns
        # Selecting specific columns from SOG
        sog_df = sog_df[['Position', 'Name', 'Salary', 'TeamAbbrev']]
        bbm_df = bbm_df[['Name', 'Minutes']]  # Selecting only Name and Minutes from BBM
        # game_logs_df: Use all columns as specified

        # Step 3: Standardize Player Names
        # Create a standardized column 'PLAYER_NAME' for all three DataFrames
        sog_df = self.standardize_player_names(sog_df, 'Name')
        sog_df.rename(columns={'Name': 'PLAYER_NAME'}, inplace=True)

        bbm_df = self.standardize_player_names(bbm_df, 'Name')
        bbm_df.rename(columns={'Name': 'PLAYER_NAME'}, inplace=True)

        game_logs_df = self.standardize_player_names(game_logs_df, 'Player')  # Assuming player names are in 'Player'
        game_logs_df.rename(columns={'Player': 'PLAYER_NAME'}, inplace=True)

        # Step 4: Merge DataFrames
        # Merge game_logs_df with sog_df and then with bbm_df on 'PLAYER_NAME'.
        merged_df = pd.merge(
            game_logs_df,  # Base DataFrame
            sog_df[['PLAYER_NAME', 'Position', 'Salary', 'TeamAbbrev']],
            on='PLAYER_NAME',
            how='left'  # Keep all rows from game_logs_df
        )

        merged_df = pd.merge(
            merged_df,
            bbm_df[['PLAYER_NAME', 'Minutes']],
            on='PLAYER_NAME',
            how='left'  # Keep all rows from the previous merge
        )

        # Step 5: Return or Save the Final Merged DataFrame
        return merged_df


    def on_data_selection_changed(self, selection):
        if selection in self.dataframes:
            self.display_dataframe(self.dataframes[selection])

    def setup_table_view(self):

        self.table_view = QTableView()
        # Better visibility style
        self.table_view.setStyleSheet("""
                    QTableView {
                        background-color: #2b2b2b;
                        gridline-color: #404040;
                        selection-background-color: #0078d7;
                        selection-color: white;
                        alternate-background-color: #333333;
                        font-size: 12px;
                        color: #e0e0e0;
                    }
                    QHeaderView::section {
                        background-color: #404040;
                        color: white;
                        padding: 5px;
                        border: 1px solid #505050;
                        font-weight: bold;
                        font-size: 12px;
                    }
                    QTableView::item {
                        padding: 5px;
                        border-bottom: 1px solid #404040;
                        color: #e0e0e0;
                    }
                """)

        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)




    def display_dataframe(self, df):
        if df is not None and not df.empty:
            try:
                print("Creating model...")  # Debug print
                model = DataFrameModel(df)

                # Create proxy model for sorting
                proxy_model = CustomSortFilterProxyModel()
                proxy_model.setSourceModel(model)

                print("Setting model to view...")  # Debug print
                self.table_view.setModel(proxy_model)

                # Enable sorting
                self.table_view.setSortingEnabled(True)

                # Adjust column widths
                for i in range(len(df.columns)):
                    self.table_view.resizeColumnToContents(i)

                print("Model set successfully")  # Debug print
            except Exception as e:
                print(f"Error displaying dataframe: {e}")

    def expand_game_logs(self, progress_print=print):
        predictions = NbaMInutesPredictions()
        predictions.nba_enhance_game_logs()
        self.merge_three_dataframes()
        #predictions.process_game_logs()
        print('Completed')
        progress_print("Done expanding game logs.")

    def import_team_stats(self, progress_print=print):
        team_abbr_dict = {
            'Atlanta Hawks': 'ATL',
            'Boston Celtics': 'BOS',
            'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL',
            'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW',
            'Houston Rockets': 'HOU',
            'Indiana Pacers': 'IND',
            'LA Clippers': 'LAC',
            'Los Angeles Lakers': 'LAL',
            'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL',
            'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL',
            'Philadelphia 76ers': 'PHI',
            'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC',
            'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA',
            'Washington Wizards': 'WAS'
        }

        try:
            progress_print("Fetching team advanced stats data...")
            season = '2024-25'
            season_type = 'Regular Season'
            df = self.fetch_advanced_team_stats(season, season_type)

            team_stats_advanced_file = self.import_folder / "advanced.csv"
            df.to_csv(team_stats_advanced_file, index=False)

            if not team_stats_advanced_file.exists():
                raise FileNotFoundError(f"team_stats_advanced.csv file not found at {team_stats_advanced_file}")

            progress_print(f"Reading file: {team_stats_advanced_file}")
            data_advanced = pd.read_csv(team_stats_advanced_file)

            progress_print("Fetching traditional stats data...")
            season = '2024-25'
            season_type = 'Regular Season'

            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star=season_type,
                measure_type_detailed_defense='Base',
                per_mode_detailed='Totals'
            )
            df = team_stats.get_data_frames()[0]

            team_stats_basic_file = self.import_folder / "traditional.csv"
            df.to_csv(team_stats_basic_file, index=False)

            if not team_stats_basic_file.exists():
                raise FileNotFoundError(f"team_stats_traditional.csv file not found at {team_stats_basic_file}")

            progress_print(f"Reading file: {team_stats_basic_file}")
            data_traditional = pd.read_csv(team_stats_basic_file)

            # After reading both CSV files, add this code:
            merged_df = pd.merge(
                data_advanced,
                data_traditional,
                on='TEAM_ID',  # Replace with the actual common column name
                how='inner'  # You can use 'left', 'right', or 'outer' depending on your needs
            )

            # Then update the rest of the code to use merged_df instead of data
            if merged_df.empty:
                raise ValueError("No data was read from the merged Team Stats CSV files")

            progress_print(f"Successfully read {len(merged_df)} rows of merged data")

            # Add the Abbr column based on the Team column
            merged_df['Abbr'] = merged_df['TEAM_NAME_x'].map(team_abbr_dict)

            # Store in dataframes dictionary
            self.dataframes['Team Stats'] = merged_df

            # Update display if Team Stats is currently selected
            if self.data_selector.currentText() == 'Team Stats':
                self.display_dataframe(merged_df)

            progress_print("Team Stats import completed successfully")
            return merged_df

        except Exception as e:
            progress_print(f"Error in Tram Stats Advanced import: {str(e)}")
            raise


    def import_bbm(self, progress_print=print):
        try:
            progress_print("Starting BBM import...")

            # Use the correct file path
            bbm_file = self.import_folder / "bbm.csv"

            if not bbm_file.exists():
                raise FileNotFoundError(f"bbm.csv file not found at {bbm_file}")

            progress_print(f"Reading file: {bbm_file}")
            data = pd.read_csv(bbm_file, index_col=False)

            # Verify we have data
            if data.empty:
                raise ValueError("No data was read from the BBM CSV file")

            progress_print(f"Successfully read {len(data)} rows of data")

            # Combine first_name and last_name into a new column called 'player'
            if 'first_name' in data.columns and 'last_name' in data.columns:
                data['Player'] = data['first_name'].str.strip() + ' ' + data['last_name'].str.strip()
                progress_print("Successfully combined first_name and last_name into 'Player' column")
                data = self.standardize_player_names(data, 'Player')
                data.rename(columns={'Player': 'PLAYER_NAME'}, inplace=True)
            else:
                raise KeyError("'first_name' or 'last_name' column is missing in the CSV file")

            # Add BB_PROJECTION column based on the formula
            required_columns = ['id', 'points', 'threes', 'rebounds', 'assists', 'steals',
                                'blocks', 'turnovers', 'double doubles', 'triple doubles']
            if all(col in data.columns for col in required_columns):
                data['BB_PROJECTION'] = data.apply(
                    lambda row: 0 if pd.isna(row['id']) or row['id'] == "" else (
                            row['points'] +
                            row['threes'] * 0.5 +
                            row['rebounds'] * 1.25 +
                            row['assists'] * 1.5 +
                            row['steals'] * 2 +
                            row['blocks'] * 2 -
                            row['turnovers'] * 0.5 +
                            row['double doubles'] * 1.5 +
                            row['triple doubles'] * 3
                    ),
                    axis=1
                )
                progress_print("Successfully added 'BB_PROJECTION' column")
            else:
                missing_cols = [col for col in required_columns if col not in data.columns]
                raise KeyError(f"The following required columns are missing in the CSV file: {missing_cols}")




            # Store in dataframes dictionary
            self.dataframes['BBM'] = data

            # Update display if BBM is currently selected
            if self.data_selector.currentText() == 'BBM':
                self.display_dataframe(data)

            progress_print("BBM import completed successfully")
            return data

        except Exception as e:
            progress_print(f"Error in BBM import: {str(e)}")
            raise

    def import_fta_entries(self, progress_print=print):
        try:
            progress_print("Starting FTA import...")

            # Use the correct file path
            fta_file = self.import_folder / "fta.csv"

            if not fta_file.exists():
                raise FileNotFoundError(f"fta.csv file not found at {fta_file}")

            progress_print(f"Reading file: {fta_file}")
            data = pd.read_csv(fta_file)

            # Verify we have data
            if data.empty:
                raise ValueError("No data was read from the FTA CSV file")

            progress_print(f"Successfully read {len(data)} rows of data")

            data = self.standardize_player_names(data, 'Name')
            data.rename(columns={'Name': 'PLAYER_NAME'}, inplace=True)


            # Store in dataframes dictionary
            self.dataframes['FTA'] = data

            # Update display if FTA is currently selected
            if self.data_selector.currentText() == 'FTA':
                self.display_dataframe(data)

            progress_print("FTA import completed successfully")
            return data

        except Exception as e:
            progress_print(f"Error in FTA import: {str(e)}")
            raise


    def import_darko(self, progress_print=print):
        try:
            progress_print("Starting Darko import...")

            # Use the correct file path
            darko_file = self.import_folder / "darko.csv"

            if not darko_file.exists():
                raise FileNotFoundError(f"darko.csv file not found at {darko_file}")

            progress_print(f"Reading file: {darko_file}")
            data = pd.read_csv(darko_file)

            # Verify we have data
            if data.empty:
                raise ValueError("No data was read from the Darko CSV file")

            progress_print(f"Successfully read {len(data)} rows of data")

            data = self.standardize_player_names(data, 'Player')
            data.rename(columns={'Player': 'PLAYER_NAME'}, inplace=True)

            # Store in dataframes dictionary
            self.dataframes['Darko'] = data

            # Update display if Darko is currently selected
            if self.data_selector.currentText() == 'Darko':
                self.display_dataframe(data)

            progress_print("Darko import completed successfully")
            return data

        except Exception as e:
            progress_print(f"Error in Darko import: {str(e)}")
            raise



    def import_dk_entries(self, progress_print=print):
        try:
            progress_print("Importing DK Entries...")

            # Set up file paths
            entries_file = self.import_folder / "entries.csv"
            output_csv_file = self.data_folder / "DKSalaries.csv"

            if not entries_file.exists():
                raise FileNotFoundError(f"entries.csv file not found at {entries_file}")

            # Read CSV data with specific columns
            progress_print("Reading CSV data...")
            columns_to_read = list(range(13, 22))
            try:
                data = pd.read_csv(
                    entries_file,
                    skiprows=7,
                    usecols=columns_to_read,
                    header=0
                )
            except Exception as e:
                raise ValueError(f"Error reading CSV: {str(e)}")

            # Verify we have data
            if data.empty:
                raise ValueError("No data was read from the CSV file")

            progress_print(f"Successfully read {len(data)} rows of data")

            # Write to output CSV
            progress_print("Writing data to output CSV...")
            try:
                data.to_csv(output_csv_file, index=False)
                progress_print(f"Data written to '{output_csv_file}'")
            except Exception as e:
                raise ValueError(f"Error writing output CSV: {str(e)}")

            data = self.standardize_player_names(data, 'Name')
            data.rename(columns={'Name': 'PLAYER_NAME'}, inplace=True)

            # Store in dataframes dictionary
            self.dataframes['DK Entries'] = data

            # Update display if DK Entries is currently selected
            if self.data_selector.currentText() == 'DK Entries':
                self.display_dataframe(data)

            progress_print("DK Entries import completed successfully")
            return data

        except Exception as e:
            progress_print(f"Error in DK Entries import: {str(e)}")
            raise

    # Fetch advanced team stats for the regular season
    def fetch_advanced_team_stats(self, season='2023-24', season_type='Regular Season'):
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Advanced'
        )
        return team_stats.get_data_frames()[0]


    from datetime import datetime, timezone, timedelta
    def fetch_and_save_team_data_with_odds(self, progress_print=print):

        team_abbr_dict = {
            'Atlanta Hawks': 'ATL',
            'Boston Celtics': 'BOS',
            'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL',
            'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW',
            'Houston Rockets': 'HOU',
            'Indiana Pacers': 'IND',
            'Los Angeles Clippers': 'LAC',
            'Los Angeles Lakers': 'LAL',
            'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL',
            'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL',
            'Philadelphia 76ers': 'PHI',
            'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC',
            'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA',
            'Washington Wizards': 'WAS'
        }

        try:
            # Get today's date in EST
            today_est = datetime.now(timezone(timedelta(hours=-5))).date()

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

            # Filter odds_data for today's games only
            today_odds_data = []
            for odds_event in odds_data:
                game_time_utc = datetime.fromisoformat(odds_event["commence_time"].replace("Z", "+00:00"))
                game_time_est = game_time_utc.astimezone(timezone(timedelta(hours=-5)))
                if game_time_est.date() == today_est:
                    today_odds_data.append(odds_event)

            progress_print("Processing team data...")
            # Prepare team data from events
            teams = []
            for event in events_data:
                # Parse game start time and convert to EST
                game_time_utc = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
                game_time_est = game_time_utc.astimezone(timezone(timedelta(hours=-5)))  # EST is UTC-5

                # Only include games that are happening today
                if game_time_est.date() == today_est:
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

            # If no games today, return early
            if not teams:
                progress_print("No games scheduled for today.")
                return

            progress_print("Creating DataFrame...")
            # Convert teams data to DataFrame
            teams_df = pd.DataFrame(teams)

            progress_print("Processing odds data...")
            # If the CSV already exists, read its content to preserve existing values
            csv_path = self.import_folder / "odds.csv"

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
            for odds_event in today_odds_data:  # Use filtered odds data
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

                # Add the Abbr column based on the Team column
                teams_df['Abbr'] = teams_df['Team'].map(team_abbr_dict)

                teams_df['Points'] = np.where(
                    (teams_df['Total'] / 2 - teams_df['Spread'] / 2) > 0,
                    teams_df['Total'] / 2 - teams_df['Spread'] / 2,
                    ''
                )


            progress_print("Saving to CSV...")
            # Save the updated DataFrame to a CSV file
            teams_df.to_csv(csv_path, index=False)

            # Store in dataframes dictionary
            self.dataframes['Odds'] = teams_df

            # Update display if DK Entries is currently selected
            if self.data_selector.currentText() == 'Odds':
                self.display_dataframe(teams_df)

            progress_print("DK Entries import completed successfully")
            return teams_df


            # Print success messages
            print("Fetched NBA Team Data with Odds:")
            print(teams_df)
            print(f"Data successfully saved to {csv_path}")

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
        self.import_darko(progress_print=progress_print)
        self.import_team_stats(progress_print=progress_print)
        progress_print("All imports completed.")

    # Post-process predictions to apply the 1% ownership cap for low minutes players
    def apply_low_minutes_cap(self, predictions, df):
        capped_predictions = predictions.copy()
        capped_predictions[df['Low_Minutes_Flag'] == 1] = np.minimum(
            capped_predictions[df['Low_Minutes_Flag'] == 1], 0.01)
        return capped_predictions

    def read_excel_data_alternative(self, sheet):
        # Try multiple methods to read the data
        methods = [
            lambda: sheet.range('A1').expand().value,
            lambda: sheet.used_range.value,
            lambda: sheet.range('A1').current_region.value
        ]

        for method in methods:
            try:
                data = method()
                if data:
                    return data
            except:
                continue

        raise ValueError("All methods failed to read Excel data")


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
                                                        num_samples=10000, print_every=2000,
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
                        if i % print_every == 0:
                            # pass
                            progress_print(f"Processing lineup {i}/{num_samples}")


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

            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, '..', 'src', 'final_nba_model.pkl')
            with open(model_path, 'rb') as file:
                model = pickle.load(file)

            file_path = os.path.join('..', 'dk_import', 'nba.xlsm')
            app = xw.App(visible=False)
            wb = app.books.open(file_path)
            sheet = wb.sheets["ownership"]

            progress_print("Reading and processing data...")

            data = self.read_excel_data_alternative(sheet)

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

            # ... your data cleaning code ...

            progress_print("Calculating weighted feasibility scores...")

            progress_print("Calculating feasibility scores...")
            feasibility = simulate_feasibility_with_progress(
                df, max_salary=1000000, lineup_size=8, num_samples=100000,
                print_every=10000, progress_print=progress_print)

            weighted_feasibility, tournament_feasibility = simulate_weighted_feasibility_with_progress(
                df, max_salary=50000, lineup_size=8, num_samples=10000,
                print_every=2000, progress_print=progress_print)


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

            shift_start = .40
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
            adjusted_predictions = rescale_with_bounds(predictions, target_sd=target_sd, top_boost=1.25)

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
    window.run_all_imports()
    window.show()
    sys.exit(app.exec())


