import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os


import numpy as np
import requests
import unicodedata

# In your MainApp class, add these imports at the top:
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from nba_api.stats.endpoints import leaguedashteamstats


from daily_download import DailyDownload
from nba_data_manager import NBADataManager
from nba_minutes_prediction_setup import NbaMInutesPredictions
from nba_minutes_predictions_enhanced import PredictMinutes
from config import NBA_CONSTANTS
from PyQt6.QtCore import Qt, QAbstractTableModel
from PyQt6.QtWidgets import QAbstractItemView



from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget, QLabel, QMessageBox, QTextEdit, QProgressDialog, QHBoxLayout,
                             QTableView, QSplitter, QComboBox, QGroupBox, QAbstractItemView)
from PyQt6.QtCore import QAbstractTableModel, Qt
import pandas as pd

from PyQt6.QtCore import QSortFilterProxyModel


class CustomSortFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.editable_columns = ['Max Minutes']  # Add any other columns you want to be editable

    def lessThan(self, left, right):
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.EditRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.EditRole)

        try:
            # Try numeric comparison first
            return float(left_data) < float(right_data)
        except (ValueError, TypeError):
            # Fall back to string comparison if numeric fails
            return str(left_data) < str(right_data)

    def flags(self, index):
        flags = super().flags(index)
        # Make specified columns editable
        column_name = self.sourceModel().headerData(index.column(), Qt.Orientation.Horizontal,
                                                    Qt.ItemDataRole.DisplayRole)
        if column_name in self.editable_columns:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role == Qt.ItemDataRole.EditRole:
            # Get the column name
            column_name = self.sourceModel().headerData(index.column(), Qt.Orientation.Horizontal,
                                                        Qt.ItemDataRole.DisplayRole)

            if column_name in self.editable_columns:
                try:
                    # Convert to float for numeric columns
                    value = float(value)
                    success = self.sourceModel().setData(self.mapToSource(index), value, role)

                    if success:
                        # Get the DataFrame from your main class
                        # You'll need to set up a way to access the DataFrame
                        # This might require storing a reference to your main class
                        # or emitting a signal to handle the save
                        self.dataChanged.emit(index, index)
                        return True
                except ValueError:
                    return False
        return False

class DataFrameModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

        return None

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None

    def setData(self, index, value, role):
        if role == Qt.ItemDataRole.EditRole:
            row = index.row()
            col = index.column()
            column_name = self._data.columns[col]
            self._data.iloc[row, col] = value
            self.dataChanged.emit(index, index)
            return True
        return False

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
        self.data_selector.addItems(['BBM', 'FTA', 'DK Entries', 'Darko', 'Team Stats', 'Odds', 'SOG', 'Predict Minutes', 'Ownership Projections', 'Export Projections'])
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

        game_logs_button = QPushButton("Merge Daily Game Logs")
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
                #self.dataframes[current_selection] = data
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

        # Dictionary for special capitalization cases
        special_capitalization = {
            # Mc/Mac names
            "mcdaniels": "McDaniels",
            "mcconnell": "McConnell",
            "mccollum": "McCollum",
            "mcgee": "McGee",
            "mcdermott": "McDermott",
            "mclemore": "McLemore",

            # De/Le names
            "derozan": "DeRozan",
            "deandre": "DeAndre",
            "demarcus": "DeMarcus",
            "deaaron": "DeAaron",
            "devonte": "DeVonte",
            "deanthony": "DeAnthony",
            "deangelis": "DeAngelis",
            "dejounte": "DeJounte",
            "demarre": "DeMarre",
            "derozan": "DeRozan",

            # La/Le names
            "lavine": "LaVine",
            "lebron": "LeBron",
            "levert": "LeVert",
            "lamelo": "LaMelo",
            "lonzo": "Lonzo",
            "luca": "Luca",

            # D' names
            "dangelo": "D'Angelo",

            # Other special cases
            "ayton": "Ayton",
            "okoro": "Okoro",
            "okogie": "Okogie",
            "anunoby": "Anunoby",
            "dosunmu": "Dosunmu"
        }

        # Common name mappings for different variations
        name_mappings = {
            "Oliviermaxence Prosper": "Olivier-maxence Prosper",
            "Herb Jones": "Herbert Jones",
            "GG Jackson": "Gregory Jackson",
            "G.G. Jackson": "Gregory Jackson",
            "Alexandre Sarr": "Alex Sarr",
            "Yongxi Cui": "Cui Yongxi",
            "Nicolas Claxton": "Nic Claxton",
            "Cameron Johnson": "Cam Johnson",
            "Kenyon Martin Jr": "KJ Martin",
            "Ron Holland": "Ronald Holland",
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
            "Kristaps Porzingis": "Kristaps Porzingis",
            "PJ Washington": "P.J. Washington",
            "RJ Barrett": "R.J. Barrett",
            "OG Anunoby": "O.G. Anunoby",
            "AJ Griffin": "A.J. Griffin",
            "Alperen Sengun": "Alperen Sengun",
            "Dennis Schroder": "Dennis Schroder",
            "Dennis Schröder": "Dennis Schroder"
        }

        def fix_internal_caps(name):
            if not isinstance(name, str):
                return name

            # Split the name into parts
            name_parts = name.split()

            # Process each part of the name
            fixed_parts = []
            for part in name_parts:
                part_lower = part.lower()
                # Check if this part needs special capitalization
                if part_lower in special_capitalization:
                    fixed_parts.append(special_capitalization[part_lower])
                else:
                    # Default capitalization (first letter capital)
                    fixed_parts.append(part.capitalize())

            return " ".join(fixed_parts)

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

        # Apply special capitalization fixes
        df[name_column] = df[name_column].apply(fix_internal_caps)

        return df

    def merge_latest_records_with_columns(self, df, merge_key='PLAYER_NAME', date_column="GAME_DATE"):
        # Convert date column to datetime if it's not already
        #df[date_column] = pd.to_datetime(df[date_column])

        # Sort by date in descending order and get the first record for each player
        latest_records = (df.sort_values(date_column, ascending=False)
                          .groupby(merge_key, as_index=False)
                          .first())

        # Fill any NaN values with 0
        latest_records = latest_records.fillna(0)

        return latest_records




    def on_data_selection_changed(self, selection):
        if selection in self.dataframes:
            print(selection)
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

    import pandas as pd
    def transform_dataframe(self, df, key):
        transformations = {
            'FTA': {
                'columns_to_remove': ['min', 'max', 'lock', 'exclude'],
                'rename_mapping': {'PLAYER_NAME': 'Player Name'},
            },
            'BBM': {
                'columns_to_remove': ['id', 'last_name', 'first_name'],
                'rename_mapping': {'BB_PROJECTION': 'BBM Projection', 'PLAYER_NAME': 'Player'},
                'move_to_front': 'Player'
            },
            'SOG': {
                'columns_to_remove': ['ID', 'Name + ID'],
                'rename_mapping': {'PLAYER_NAME': 'Player'},
                'move_to_front': 'Player'
            },
            'Team Stats': {
                'columns_to_remove': ['TEAM_ID', 'last_name', 'first_name'],
                'rename_mapping': {'TEAM_NAME_x': 'Team', 'W_PCT_x': 'Win Pct'},
                'wildcard_remove': 'RANK',  # Indicate we want to remove columns with 'RANK' in column names
            },
        }

        def remove_duplicate_columns(df):
            base_columns = set(col.split('_')[0] for col in df.columns if col.endswith(('_x', '_y')))

            for base in base_columns:
                print(f"Processing base column: {base}")
                col_without_suffix = base
                col_x = f"{base}_x"
                col_y = f"{base}_y"

                if col_without_suffix in df.columns:
                    df = df.drop(columns=[col_x, col_y], errors='ignore')
                elif col_x in df.columns and col_y in df.columns:
                    df = df.drop(columns=[col_y], errors='ignore')
                    df = df.rename(columns={col_x: base})
                elif col_x in df.columns:
                    df = df.rename(columns={col_x: base})
                elif col_y in df.columns:
                    df = df.rename(columns={col_y: base})

            #print(f"Final columns: {df.columns}")
            return df


        # Handle wildcard column removal
        def remove_columns_by_wildcard(df, keyword):
            cols_to_remove = [col for col in df.columns if keyword in col]
            return df.drop(columns=cols_to_remove, errors='ignore')

        if key in transformations:
            config = transformations[key]

            # Remove specified columns
            if 'columns_to_remove' in config:
                df = df.drop(columns=config['columns_to_remove'], axis=1, errors='ignore')  # Safe removal

            # Rename columns
            if 'rename_mapping' in config:
                df = df.rename(columns=config['rename_mapping'])

            # Handle duplicate columns based on suffixes
            df = remove_duplicate_columns(df)

            # Handle wildcard column removal
            if 'wildcard_remove' in config:
                df = remove_columns_by_wildcard(df, config['wildcard_remove'])

            # Reorder columns
            if 'columns_reorder' in config:
                reorder = [col for col in config['columns_reorder'] if
                           col in df.columns]  # Retain only existing columns
                df = df[reorder]  # Reorder based on the specified list

            # Move a specific column to the front
            if 'move_to_front' in config:
                column_to_move = config['move_to_front']
                if column_to_move in df.columns:
                    columns = [column_to_move] + [col for col in df.columns if col != column_to_move]
                    df = df[columns]

        return df

    # Adjusted adjust_column_widths method for PyQt6
    def adjust_column_widths(self):
        header = self.table_view.horizontalHeader()
        model = self.table_view.model()
        column_count = model.columnCount()

        for column in range(column_count):
            # Get the width needed for the header text
            header_width = header.fontMetrics().boundingRect(
                model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            ).width() + 80  # Add some padding

            # First resize to content
            self.table_view.resizeColumnToContents(column)
            # Get content-based width
            content_width = self.table_view.columnWidth(column)

            # Use the maximum of header width and content width
            self.table_view.setColumnWidth(column, max(header_width, content_width))

    def display_dataframe(self, df, sort_column=None, ascending=False):
        if df is not None and not df.empty:
            try:
                print("Creating model...", len(df.columns))  # Debug print
                print("DARKO LOADED", df.equals(self.dataframes['Darko']))

                if sort_column:
                    if sort_column in df.columns:
                        print(f"Sorting DataFrame by column: {sort_column}, ascending: {ascending}")  # Debug print
                        df = df.sort_values(by=sort_column, ascending=ascending)
                    else:
                        print(
                            f"Warning: The specified sort_column '{sort_column}' is not in DataFrame columns.")  # Debug warning


                # Apply transformations based on DataFrame key
                if df.equals(self.dataframes['FTA']):
                    df = self.transform_dataframe(df, key='FTA')
                elif df.equals(self.dataframes['BBM']):
                    df = self.transform_dataframe(df, key='BBM')
                elif df.equals(self.dataframes['Team Stats']):
                    df = self.transform_dataframe(df, key='Team Stats')
                elif df.equals(self.dataframes['SOG']):
                    df = self.transform_dataframe(df, key='SOG')


                model = DataFrameModel(df)
                # Create proxy model for sorting
                proxy_model = CustomSortFilterProxyModel()
                proxy_model.setSourceModel(model)

                print("Setting model to view...")  # Debug print
                self.table_view.setModel(proxy_model)
                # Enable sorting and editing
                self.table_view.setSortingEnabled(True)

                # Optional: Set initial sort if sort_column is specified and verified
                if sort_column and sort_column in df.columns:
                    column_index = df.columns.get_loc(sort_column)
                    sort_order = Qt.SortOrder.AscendingOrder if ascending else Qt.SortOrder.DescendingOrder
                    self.table_view.sortByColumn(column_index, sort_order)

                self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)

                # Connect to handle data changes
                proxy_model.dataChanged.connect(self.handle_data_changed)

                # Adjust column widths
                self.adjust_column_widths()



                # Make last column stretch to fill remaining space
                #self.table_view.horizontalHeader().setStretchLastSection(True)


                print("Model set successfully")  # Debug print
            except Exception as e:
                print(f"Error displaying dataframe: {e}")

    def handle_data_changed(self, topLeft, bottomRight):
        try:
            # Get the current DataFrame
            current_df = self.dataframes[self.data_selector.currentText()]

            # Save to CSV
            csv_path = os.path.join('..', 'dk_import', 'nba_daily_combined.csv')
            current_df.to_csv(csv_path, index=False)
            print("Data saved successfully")
        except Exception as e:
            print(f"Error saving data: {e}")

    def expand_game_logs(self, progress_print=print):
        predictions = NbaMInutesPredictions()
        predictions.nba_enhance_game_logs()
        self.process_game_logs()
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
            #if self.data_selector.currentText() == 'Team Stats':
            self.display_dataframe(merged_df)

            progress_print("Team Stats import completed successfully")
            return merged_df

        except Exception as e:
            progress_print(f"Error in Tram Stats Advanced import: {str(e)}")
            raise



    def merge_dataframes_sog(self, progress_print=print):
        """
        Merges DK, BBM, FTA, DARKO and game logs DataFrames based on standardized player names.
        """
        # Step 1: Load the DataFrames
        dk_df = self.dataframes['DK Entries']
        bbm_df = self.dataframes['BBM']
        fta_df = self.dataframes['FTA']
        darko_df = self.dataframes['Darko']
        own_df = self.dataframes['Ownership Projections']

        bbm_df = bbm_df[['PLAYER_NAME', 'BB_PROJECTION']]  # Selecting only Name and Minutes from BBM
        fta_df = fta_df[['PLAYER_NAME', 'Minutes', 'Projection','Ownership']]  # Selecting only Name and Minutes from BBM

        darko_df = darko_df[['PLAYER_NAME', 'minutes',
                             'pts', 'reb', 'ast', 'stl', 'blk', 'tov',
                             'dk', 'dk_rate', 'dk_usg_rate', 'usg_G',
                             'Defense', 'SD_Score',
                             ]]  # Selecting only Name and Minutes from BBM

        own_df = own_df[['DK Name', 'Predicted Ownership']]
        # Step 4: Merge DataFrames
        # Merge game_logs_df with sog_df and then with bbm_df on 'PLAYER_NAME'.
        data = pd.merge(
            dk_df,  # Base DataFrame
            bbm_df,
            on='PLAYER_NAME',
            how='left'  # Keep all rows from game_logs_df
        )

        data = pd.merge(
             data,
             fta_df,
             on='PLAYER_NAME',
             how='left'  # Keep all rows from the previous merge
        )

        data = pd.merge(
             data,
             darko_df,
             on='PLAYER_NAME',
             how='left'  # Keep all rows from the previous merge
        )



        data = pd.merge(
            data,
            own_df,
            left_on='PLAYER_NAME',
            right_on='DK Name',
            how='left'  # Keep all rows from the previous merge
        )

        # Verify we have data
        if data.empty:
            raise ValueError("No data was read from the SOG CSV file")

        progress_print(f"Successfully read {len(data)} rows of data")
        #
        # data.rename(columns={'minutes': 'BB Minutes'}, inplace=True)
        # data.rename(columns={'Minutes': 'FTA Minutes'}, inplace=True)
        # data.rename(columns={'BB_PROJECTION': 'BB Projection'}, inplace=True)
        # data.rename(columns={'Projection': 'FTA Projection'}, inplace=True)


        # Store in dataframes dictionary
        self.dataframes['SOG'] = data

        # Update display if FTA is currently selected
        # if self.data_selector.currentText() == 'SOG':
        #     self.display_dataframe(data)

        progress_print("SOG import completed successfully")
        return data


    def process_game_logs(self):
        # Read from Excel using XWings

        time.sleep(1)  # Give Excel a moment to fully initialize
        # Connect to Excel
        # excel_path = os.path.join('..', 'dk_import', 'nba - Copy.xlsm')
        csv_path = os.path.join('..', 'dk_import', 'nba_daily_combined.csv')
        csv_read = os.path.join('..', 'dk_import', 'nba_boxscores_enhanced.csv')

        # Usage:
        df = pd.read_csv(csv_read, encoding='utf-8')
        result_df = self.merge_latest_records_with_columns(df)  # using default parameters
        result_df = result_df.rename(columns={'Projection': 'Projected Pts'})

        dk_df = self.dataframes['DK Entries']
        bbm_df = self.dataframes['BBM']
        game_logs_df = result_df

        #dk_df.rename(columns={'PLAYER_NAME': 'Player'}, inplace=True)
        #game_logs_df.rename(columns={'PLAYER_NAME': 'Player'}, inplace=True)
        game_logs_df.rename(columns={'TEAM_NAME': 'Team'}, inplace=True)
        game_logs_df.rename(columns={'MIN': 'Minutes Last Game'}, inplace=True)
        #bbm_df.rename(columns={'PLAYER_NAME': 'Player'}, inplace=True)
        bbm_df.rename(columns={'BB_PROJECTION': 'Projection'}, inplace=True)
        bbm_df.rename(columns={'minutes': 'Minutes'}, inplace=True)

        game_logs_df = self.standardize_player_names(game_logs_df, 'PLAYER_NAME')
        bbm_df = self.standardize_player_names(bbm_df, 'PLAYER_NAME')

        data = pd.merge(
            dk_df,
            game_logs_df,
            on='PLAYER_NAME',
            how='left'  # Keep all rows from the previous merge
        )

        data = pd.merge(
            data,
            bbm_df,
            on='PLAYER_NAME',
            how='left'  # Keep all rows from the previous merge
        )

        # Fill NaN values based on column type
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        string_columns = data.select_dtypes(include=['object']).columns

        # Fill numeric columns with 0
        data[numeric_columns] = data[numeric_columns].fillna(0)
        # Fill string columns with empty string
        data[string_columns] = data[string_columns].fillna('')



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
        # result_df = result_df[new_column_order]

        # After reading the CSV and before displaying:
        if 'Max Minutes' not in data.columns:
            data['Max Minutes'] = 48 # Initialize with current minutes

        #data = result_df

        # Verify we have data
        if data.empty:
            raise ValueError("No data was read from the FTA CSV file")

        print(f"Successfully read {len(data)} rows of data")

        game_logs_df.rename(columns={'PLAYER_NAME': 'Player'}, inplace=True)
        #data = self.standardize_player_names(data, 'Player')

        # Drop columns that end with '_x' or '_y' (default merge suffixes)
        data = data.loc[:, ~data.columns.str.contains('_x|_y$', regex=True)]

        # Or drop specific duplicate columns
        duplicate_cols = data.columns[data.columns.duplicated()]
        data = data.drop(columns=duplicate_cols)

        data = data.fillna('')
        data.to_csv(csv_path, index=False)



        # Store in dataframes dictionary
        self.dataframes['Game Logs'] = data
        # If you're using a model-view setup, you'll need to set up an editable model:
        #if self.data_selector.currentText() == 'Game Logs':
        self.display_dataframe(data)

        # if self.data_selector.currentText() == 'Game Logs':
        #     self.display_dataframe(data)

        return data


    def build_predict_minutes_dataframe(self, progress_print=print):
        time.sleep(1)  # Give Excel a moment to fully initialize
        csv_path = os.path.join('..', 'dk_import', 'nba_daily_combined.csv')
        game_logs_df = pd.read_csv(csv_path, encoding='utf-8', keep_default_na=False)
        game_logs_df['injury'] = game_logs_df['injury'].replace("", "Active")
        game_logs_df.rename(columns={'Player': 'PLAYER_NAME'}, inplace=True)
        game_logs_df['Original_Minutes'] = game_logs_df['Minutes'].round(2)
        game_logs_df['Predicted_Minutes'] = game_logs_df['Minutes']
        self.dataframes['Predict Minutes'] = game_logs_df
        self.display_dataframe(game_logs_df)




    def predict_minutes(self, progress_print=print):
        data = self.Predictor(progress_print)
        return data

    def Predictor(self, progress_print):
        predictions = PredictMinutes()
        game_logs_df = self.dataframes['Predict Minutes']
        game_logs_df.rename(columns={'PLAYER_NAME': 'Player'}, inplace=True)
        data = predictions.predict_minutes_df(game_logs_df)
        data.rename(columns={'Player': 'PLAYER_NAME'}, inplace=True)

        data['Original_Minutes'] = data['Original_Minutes'].round(2)
        self.dataframes['Predict Minutes'] = data
        self.update_darko(data)
        self.merge_dataframes_sog()
        # self.display_dataframe(data)
        progress_print("Predict Minutes import completed successfully")
        return data

    def import_bbm(self, progress_print=print):
        try:
            progress_print("Starting BBM import...")

            # Use the correct file path
            bbm_file = self.import_folder / "bbm.csv"

            if not bbm_file.exists():
                raise FileNotFoundError(f"bbm.csv file not found at {bbm_file}")

            progress_print(f"Reading file: {bbm_file}")
            data = pd.read_csv(bbm_file, index_col=False, encoding='utf-8', keep_default_na=False)

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

            # Round numeric columns to 2 or 3 decimal places
            numeric_cols = ['points', 'threes', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers',
                            'BB_PROJECTION']
            data[numeric_cols] = data[numeric_cols].round(2)  # Adjust precision here
            data = data.fillna('')

            self.dataframes['BBM'] = data
            #self.display_dataframe(data)
            progress_print("BBM import completed successfully")
            return data

        except Exception as e:
            progress_print(f"Error in BBM import: {str(e)}")
            raise

    def import_fta_entries(self, progress_print=print):
        try:
            progress_print("Starting FTA import...")
            fta_file = self.import_folder / "fta.csv"
            if not fta_file.exists():
                raise FileNotFoundError(f"fta.csv file not found at {fta_file}")

            progress_print(f"Reading file: {fta_file}")

            # Read file
            data = pd.read_csv(fta_file, encoding='utf-8', keep_default_na=False)
            if data.empty:
                raise ValueError("No data was read from the FTA CSV file")
            progress_print(f"Successfully read {len(data)} rows of data")

            # Standardize and rename columns
            data = self.standardize_player_names(data, 'Name')
            data.rename(columns={'Name': 'PLAYER_NAME'}, inplace=True)  # Rename Name -> PLAYER_NAME

            # Store raw data (in case it is needed elsewhere)
            self.dataframes['FTA'] = data

            # Remove unwanted columns and create DataFrame for display
            columns_to_remove = ['min', 'max', 'lock', 'exclude']
            display_data = data.drop(columns=columns_to_remove, axis=1)  # Remove columns

            # Rename PLAYER_NAME -> Player Name for display purposes
            display_data = display_data.rename(columns={'PLAYER_NAME': 'Player Name'})

            # Pass the correct display_data to the display_dataframe method
            #self.display_dataframe(display_data)

            progress_print("FTA import completed successfully")
            return data

        except Exception as e:
            progress_print(f"Error in FTA import: {str(e)}")
            raise

    # A dictionary mapping column names to their calculation logic
    FORMULA_CONFIG = {
        'gm2': lambda row: ((row['FGA/100'] - row['FG3A/100']) * row['FG2%'] * row['minutes']) / 48,

        # Add more column formulas here in the same way:
        # 'column_name': lambda row: YOUR_EXCEL_EQUATION_IN_PYTHON,
       # 'col2': lambda row: row['FG3A/100'] * row['FG3%'],  # Example formula.
       # 'col3': lambda row: row['FGA/100'] * row['Turnover%'] / row['minutes'] if row['minutes'] > 0 else 0,
        # Repeat for all 15 columns
    }

    def update_darko(self, df, progress_print=print):
        progress_print("Updating Darko ...")
        data = self.dataframes['Darko']

        # Verify we have data
        if data.empty:
            raise ValueError("No data was read from Darko")
        progress_print(f"Successfully read {len(data)} rows of data")



        # Extract the 'Minutes' column from BBM and merge into Darko data
        if df.equals(self.dataframes['BBM']):
            bbm_df = df
            bbm_df = self.standardize_player_names(bbm_df, 'PLAYER_NAME')  # Ensure consistent name formatting
            merged_data = pd.merge(data, bbm_df[['PLAYER_NAME', 'minutes']], on='PLAYER_NAME', how='left')
            data['minutes'] = merged_data['minutes']  # Update minutes column directly
            progress_print("Successfully merged 'Minutes' column from BBM.")
        else:
            min_df = self.dataframes['Predict Minutes']
            min_df = self.standardize_player_names(min_df, 'PLAYER_NAME')  # Ensure consistent name formatting
            merged_data = pd.merge(data, min_df[['PLAYER_NAME', 'Predicted_Minutes']], on='PLAYER_NAME', how='left')
            data['minutes'] = merged_data['Predicted_Minutes']  # Update minutes column with predicted values
            progress_print("Successfully updated 'minutes' column with predicted minutes.")
            # Apply formulas for all columns in FORMULA_CONFIG


        # Vectorized formula
        data['2gm'] =( (data['FGA/100'] - data['FG3A/100']) *data['FG2%'] * data['minutes'] / 48).round(2)
        data['3gm'] = (
                data['FG3A/100'] * data['FG3%'] * data['minutes'] / 48
        ).round(2)
        # Calculate 'ftm' column with vectorized operations and round to two decimals
        data['ftm'] = (
                data['FTA/100'] * data['FT%'] * data['minutes'] / 48
        ).round(2)
        # Calculate 'pts' using existing columns 'gm2', 'gm3', and 'ftm', and round to two decimals
        data['pts'] = (
                data['2gm'] * 2 +  # Multiply 'gm2' by 2
                data['3gm'] * 3 +  # Multiply 'gm3' by 3
                data['ftm']  # Add 'ftm'
        ).round(2)
        # Vectorized calculation of 'reb'
        data['reb'] = (
                data['REB/100'] * data['minutes'] / 48
        ).round(2)
        # Vectorized calculation of 'ast'
        data['ast'] = (
                data['AST/100'] * data['minutes'] / 48
        ).round(2)
        # Vectorized calculation of 'blk'
        data['blk'] = (
                data['BLK/100'] * data['minutes'] / 48
        ).round(2)
        # Vectorized calculation of 'stl'
        data['stl'] = (
                data['STL/100'] * data['minutes'] / 48
        ).round(2)
        # Vectorized calculation of 'tov'
        data['tov'] = (
                data['TOV/100'] * data['minutes'] / 48
        ).round(2)

        # Calculate the column conditionally as per the given logic
        data['dd'] = np.where(
            (data['pts'] >= 10) & ((data['reb'] >= 10) | (data['ast'] >= 10)),  # Condition 1
            1.5,  # Value if condition is True
            0  # Value if condition is False
        ).round(2)

        # Calculate 'dk' using the provided formula
        data['dk'] = (
                data['pts'] +  # Points
                data['3gm'] * 0.5 +  # Three-pointers made (0.5 multiplier)
                data['reb'] * 1.25 +  # Rebounds (1.25 multiplier)
                data['ast'] * 1.5 +  # Assists (1.5 multiplier)
                data['blk'] * 2 +  # Blocks (2 multiplier)
                data['stl'] * 2 -  # Steals (2 multiplier)
                data['tov'] * 0.5 +  # Turnovers (-0.5 multiplier)
                data['dd']  # Double-double bonus
        ).round(2)  # Round to two decimal places

        # Calculate 'dk_rate', handle cases where 'minutes' is 0
        data['dk_rate'] = np.where(
            data['minutes'] > 0,  # Only calculate if 'minutes' is greater than 0
            (data['dk'] / data['minutes']).round(2),  # Perform the division
            0  # Assign 0 if 'minutes' is 0
        )

        # Calculate 'dk_usg_rate', handle cases where minutes = 0
        data['dk_usg_rate'] = np.where(
            data['minutes'] > 0,  # Only calculate if minutes > 0
            ((data['pts'] + data['3gm'] * 0.5 - data['tov'] * 0.5) / data['minutes']).round(2),
            0  # Assign dk_usg_rate = 0 where minutes = 0
        )

        # Calculate 'usg G' based on the formula
        data['usg_G'] = (
                data['FGA/100'] +  # Divide FGA by 100
                data['FTA/100']  * 0.44 +  # Divide FTA by 100 and multiply by 0.44
                data['TOV/100']  # Divide TOV by 100
        ).round(2)  # Round to 4 decimal places

        # Gracefully handle missing values and cases where minutes = 0
        data['Defense'] = (
                (data['D-DPM'].fillna(0) / 48) * data['minutes'].fillna(0)
        ).round(2)

        # Assume `data` is your DataFrame containing the stats columns
        data['SD_Score'] = np.where(
            data['minutes'] > 0,  # Only calculate if minutes > 0
            (
                    data['ast'] * NBA_CONSTANTS['SD_AST'] * 1.5 +  # Assists
                    data['reb'] * NBA_CONSTANTS['SD_REB'] * 1.25 +  # Rebounds
                    data['3gm'] * NBA_CONSTANTS['SD_3GM'] * 3.5 +  # Three-pointers made
                    data['stl'] * NBA_CONSTANTS['SD_STL'] * 2 +  # Steals
                    data['blk'] * NBA_CONSTANTS['SD_BLK'] * 2 -  # Blocks
                    data['tov'] * NBA_CONSTANTS['SD_TO'] * 0.5 +  # Turnovers
                    data['2gm'] * NBA_CONSTANTS['SD_2GM'] * 2 +  # Two-point field goals made
                    data['ftm'] * NBA_CONSTANTS['SD_FTM']  # Free throws made
            ) / data['minutes'] * 0.5,  # Scale by Constant M32 (ceiling factor)
            0  # If minutes = 0, assign SD_Score = 0
        ).round(3)

        # # Optional: Reorganize output column order if needed
        # desired_order = ['nba_id', 'minutes', 'gm2', 'col2', 'col3', 'other_columns']
        # data = data[desired_order]
        print("darko columns : ", len(data.columns))

        progress_print(f"Successfully added {len(self.FORMULA_CONFIG)} new calculated columns.")
        # Update the Darko dataframe
        self.dataframes['Darko'] = data
        #self.display_dataframe(data)
        progress_print("Darko update completed successfully")


        return data

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
            #if self.data_selector.currentText() == 'Darko':

            #self.display_dataframe(data)

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

                #update odds

            progress_print("Saving to CSV...")
            # Save the updated DataFrame to a CSV file
            teams_df.to_csv(csv_path, index=False)
            progress_print("DK Entries import completed successfully")
            # Print success messages
            print("Fetched NBA Team Data with Odds:")
            print(teams_df)
            print(f"Data successfully saved to {csv_path}")
            progress_print("Done importing NBA Schedule to odds.")
            self.update_team_data_with_odds()
            return teams_df


        except Exception as e:
            progress_print(f"An error occurred: {e}")
            raise


    from datetime import datetime, timezone, timedelta
    def update_team_data_with_odds(self, progress_print=print):

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

            progress_print("Processing odds data...")
            # If the CSV already exists, read its content to preserve existing values
            csv_path = self.import_folder / "odds.csv"

            if os.path.exists(csv_path):
                existing_data = pd.read_csv(csv_path)
                teams_df = pd.read_csv(csv_path, index_col=False, encoding='utf-8', keep_default_na=False)

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

            # Add the Abbr column based on the Team column
            teams_df['Abbr'] = teams_df['Team'].map(team_abbr_dict)

            teams_df['Points'] = np.where(
                (teams_df['Total'] / 2 - teams_df['Spread'] / 2) > 0,
                teams_df['Total'] / 2 - teams_df['Spread'] / 2,
                ''
            )



            # Store in dataframes dictionary
            self.dataframes['Odds'] = teams_df

            # Update display if DK Entries is currently selected
            #if self.data_selector.currentText() == 'Odds':
            self.display_dataframe(teams_df)

            progress_print("DK Entries import completed successfully")


            # Print success messages
            print("Fetched NBA Team Data with Odds:")
            print(teams_df)
            print(f"Data successfully saved to {csv_path}")

            progress_print("Done importing NBA Schedule to odds.")
            return teams_df


        except Exception as e:
            progress_print(f"An error occurred: {e}")
            raise

    def export_projections(self, progress_print=print):
        progress_print("Exporting projections to CSV...")
        try:
            # Set up paths
            current_folder = Path(__file__).resolve().parent
            dk_data_folder = current_folder.parent / "dk_data"
            progress_print("Creating output directory...")
            dk_data_folder.mkdir(exist_ok=True)
            output_csv_file = dk_data_folder / "projections.csv"

            progress_print("Filtering required columns from 'SOG' DataFrame...")
            # Specify keys/columns to keep
            #selected_columns = ['PLAYER_NAME', 'Position', 'Team', 'Minutes', 'Salary', 'Projection', 'Ownership', 'SD_Score', 'BB_PROJECTION']  # Replace with your desired column names
            selected_columns = ['PLAYER_NAME', 'Position', 'TeamAbbrev',
             'minutes', 'Salary', 'dk', 'Predicted Ownership', 'SD_Score', 'BB_PROJECTION'
                                ]  # Replace with your desired column names
            filtered_sog = self.dataframes['SOG'][selected_columns].copy()  # Filter the DataFrame
            numeric_columns = ['minutes', 'Salary', 'dk', 'Predicted Ownership', 'SD_Score', 'BB_PROJECTION']
            for col in numeric_columns:
                filtered_sog[col] = pd.to_numeric(filtered_sog[col], errors='coerce').fillna(0)

            renamed_columns = {
                'PLAYER_NAME': 'Name',
                'minutes': 'Minutes',
                'TeamAbbrev': 'Team',
                'dk': 'Fpts',
                'Predicted Ownership': 'Own%',
                'SD_Score': 'StdDev',
                'BB_PROJECTION': 'FieldFpts'
            }

            filtered_sog = filtered_sog.rename(columns=renamed_columns)
            self.dataframes['Export Projections'] = filtered_sog

            #
            # if self.data_selector.currentText() == 'Export Projections':
            #     self.display_dataframe(filtered_sog)

            progress_print("Exporting filtered data to CSV...")
            filtered_sog.to_csv(output_csv_file, index=False)

            progress_print(f"Projections successfully exported to '{output_csv_file}'.")
        except KeyError as e:
            progress_print(f"KeyError: Missing required column(s) - {e}")
            raise
        except Exception as e:
            progress_print(f"An error occurred: {str(e)}")
            raise

    def run_all_imports(self, progress_print=print):
        progress_print("Running all imports...")
        self.import_bbm(progress_print=progress_print)
        self.import_fta_entries(progress_print=progress_print)
        self.import_dk_entries(progress_print=progress_print)
        self.import_darko(progress_print=progress_print)
        self.update_darko(self.dataframes['BBM'], progress_print=progress_print)
        self.import_team_stats(progress_print=progress_print)
        self.build_ownership_projections_dataframe(progress_print=progress_print)
        self.merge_dataframes_sog(progress_print=progress_print)
        self.build_predict_minutes_dataframe(progress_print=progress_print)
        self.export_projections(progress_print=progress_print)
        self.update_team_data_with_odds()
        #self.export_projections(progress_print=progress_print)
        progress_print("All imports completed successfully.")

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

    def build_ownership_projections_dataframe(self, progress_print=print):
        # Step 1: Load the DataFrames
        dk_df = self.dataframes['DK Entries']
        bbm_df = self.dataframes['BBM']
        darko_df = self.dataframes['Darko']
        fta_df = self.dataframes['FTA']

        # Step 2: Start with DK entries as base and merge others
        new_df = dk_df[['PLAYER_NAME', 'Position', 'Salary']].copy()

        # Merge with BBM data
        new_df = new_df.merge(
            bbm_df[['PLAYER_NAME', 'minutes', 'BB_PROJECTION']],
            left_on='PLAYER_NAME',
            right_on='PLAYER_NAME',
            how='left'
        )



        # Merge with Darko data
        new_df = new_df.merge(
            darko_df[['PLAYER_NAME', 'Team']],
            left_on='PLAYER_NAME',
            right_on='PLAYER_NAME',
            how='left'
        )

        # Drop duplicate Player columns from merges
        new_df = new_df.drop(columns=['Player_x', 'Player_y'], errors='ignore')


        # Step 3: Rename columns
        column_renames = {
            'PLAYER_NAME': 'DK Name',
            'Position': 'Position',
            'Salary': 'Salary',
            'minutes': 'Minutes',
            'BB_PROJECTION': 'Points Proj',
            'Team': 'Team'
        }
        new_df = new_df.rename(columns=column_renames)

        # Step 4: Remove rows with any NaN values
        new_df = new_df.dropna(how='any')




        #data.rename(columns={'Name': 'PLAYER_NAME'}, inplace=True)
        # Step 6: Add calculated columns
        new_df['Value'] = (new_df['Points Proj'] * 1000) / new_df['Salary']
        new_df['Ceiling'] = new_df['Points Proj'] * 1.5
        new_df['Plus'] = new_df['Points Proj'] - new_df['Salary'] * 5 / 1000
        # Format numeric columns to 2 decimal places
        numeric_columns = ['Salary', 'Minutes', 'Points Proj', 'Value', 'Ceiling', 'Plus']
        for col in numeric_columns:
            new_df[col] = new_df[col].round(2)

        new_df = self.standardize_player_names(new_df, 'DK Name')
        new_df['Own Actual'] = 10
        new_df['Predicted Ownership'] = 10

        # Store in dataframes dictionary

        # Update display if DK Entries is currently selected
        # if self.data_selector.currentText() == 'DK Entries':
        new_df = new_df.sort_values(by='Points Proj', ascending=False)
        self.dataframes['Ownership Projections'] = new_df
        self.display_dataframe(new_df, sort_column='Points Proj', ascending=False)

        return new_df



    def ownership_projections(self, progress_print=print):
        import os
        import pandas as pd
        import pickle
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


            progress_print("Reading and processing data...")

            data = self.dataframes['Ownership Projections']

            #data = sheet.range('A1').expand().value
            df = data
            #df['Salary'] = df['Salary'].astype(int)  # Convert to integers
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

            progress_print("Calculating weighted feasibility scores...")
            progress_print("Calculating feasibility scores...")
            feasibility = simulate_feasibility_with_progress(
                df, max_salary=1000000, lineup_size=8, num_samples=100000,
                print_every=10000, progress_print=progress_print)

            weighted_feasibility, tournament_feasibility = simulate_weighted_feasibility_with_progress(
                df, max_salary=50000, lineup_size=8, num_samples=2000,
                print_every=200, progress_print=progress_print)


            # Ensure all base columns are numeric
            df['Minutes'] = pd.to_numeric(df['Minutes'], errors='coerce')
            df['Points Proj'] = pd.to_numeric(df['Points Proj'], errors='coerce')
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df['Ceiling'] = pd.to_numeric(df['Ceiling'], errors='coerce')

            df['Feasibility'] = df.apply(lambda row: feasibility.get(row['DK Name'], 0), axis=1)
            df['Weighted Feasibility'] = df.apply(lambda row: weighted_feasibility.get(row['DK Name'], 0), axis=1)
            df['Tournament Feasibility'] = df.apply(lambda row: tournament_feasibility.get(row['DK Name'], 0), axis=1)
            df['Feasibility'] = df['Feasibility'].astype(float) * df['Minutes'].astype(float)

            df['Proj_Salary_Ratio'] = df['Points Proj'] / df['Salary']
            df['Normalized_Value'] = df.groupby('Contest ID')['Value'].transform(lambda x: (x - x.mean()) / x.std())
            df['Prev_Salary_Mean'] = df.groupby('DK Name')['Salary'].shift(1).rolling(window=3).mean()
            df['Log_Proj'] = np.log1p(df['Points Proj'])

            # Add interaction features
            df['Salary_Proj_Interaction'] = df['Salary'] * df['Points Proj']
            df['Proj_Feasibility_Interaction'] = df['Points Proj'] * df['Feasibility']
            df['Value_Feasibility_Interaction'] = df['Value'] * df['Feasibility']

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

            final_predictions = final_predictions.round(2)  # Round to 2 decimal places
            self.dataframes['Ownership Projections']['Predicted Ownership'] = final_predictions

            self.merge_dataframes_sog()

            if self.data_selector.currentText() == 'Ownership Projections':
                self.display_dataframe(data)


        except Exception as e:
            progress_print(f"An error occurred: {e}")
            raise



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImportTool()
    window.run_all_imports()
    window.show()
    sys.exit(app.exec())


