import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

import numpy as np
import pandas as pd
import requests
import xlwings as xw
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox, \
    QTextEdit, QProgressDialog, QGroupBox, QHeaderView
from nba_api.stats.endpoints import leaguedashteamstats

#from nba_fetch import fetch_advanced_team_stats


# In your MainApp class, add these imports at the top:
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from daily_download import DailyDownload
from nba_data_manager import NBADataManager
from nba_minutes_prediction_setup import NbaMInutesPredictions
from nba_minutes_predictions_enhanced import PredictMinutes



from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget, QLabel, QMessageBox, QTextEdit, QProgressDialog, QHBoxLayout,
                             QTableView, QSplitter, QComboBox)
from PyQt6.QtCore import QAbstractTableModel, Qt
import pandas as pd

from PyQt6.QtCore import QSortFilterProxyModel

# class DataFrameModel(QAbstractTableModel):
#     def __init__(self, data):
#         super().__init__()
#         self._data = data
#         self._original_data = data.copy()
#
#     def rowCount(self, parent=None):
#         return len(self._data)
#
#     def columnCount(self, parent=None):
#         return len(self._data.columns)
#
#     # def data(self, index, role=Qt.ItemDataRole.DisplayRole):
#     #     if role == Qt.ItemDataRole.DisplayRole:
#     #         value = self._data.iloc[index.row(), index.column()]
#     #         return str(value)
#     #     return None
#
#     def data(self, index, role=Qt.ItemDataRole.DisplayRole):
#         if not index.isValid():
#             return None
#
#         if role == Qt.ItemDataRole.DisplayRole:
#             value = self._data.iloc[index.row(), index.column()]
#             if pd.isna(value):
#                 return ""
#             if isinstance(value, (float, np.float64)):
#                 return f"{value:.2f}"
#             return str(value)
#
#         elif role == Qt.ItemDataRole.TextAlignmentRole:
#             value = self._data.iloc[index.row(), index.column()]
#             if isinstance(value, (int, float, np.number)):
#                 # Align numbers to the right
#                 return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
#             return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
#
#         return None
#
#     def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
#         if role == Qt.ItemDataRole.DisplayRole:
#             if orientation == Qt.Orientation.Horizontal:
#                 return str(self._data.columns[section])
#             if orientation == Qt.Orientation.Vertical:
#                 return str(section + 1)
#         return None
#
#     def sort(self, column, order):
#         """Sort table by given column number."""
#         try:
#             column_name = self._data.columns[column]
#             ascending = order == Qt.SortOrder.AscendingOrder
#             self._data = self._data.sort_values(column_name, ascending=ascending)
#             self.layoutChanged.emit()
#         except Exception as e:
#             print(f"Sorting error: {e}")
#
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

        # Create the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create a splitter for buttons and table
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)


        # Left side - Buttons
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        splitter.addWidget(button_widget)

        # Add data selection combo box
        self.data_selector = QComboBox()
        self.data_selector = QComboBox()
        self.data_selector.addItems(['BBM', 'FTA', 'DK Entries', 'Darko', 'Projections'])
        self.data_selector.currentTextChanged.connect(self.on_data_selection_changed)
        button_layout.addWidget(self.data_selector)

        # Add buttons
        bbm_button = QPushButton("Import BBM")
        bbm_button.clicked.connect(lambda: self.run_threaded_import(self.import_bbm))
        button_layout.addWidget(bbm_button)

        # Add DK Entries button
        dk_entries_button = QPushButton("Import DK Entries")
        dk_entries_button.clicked.connect(lambda: self.run_threaded_import(self.import_dk_entries))
        button_layout.addWidget(dk_entries_button)

        # Add Darko button
        darko_button = QPushButton("Import Darko")
        darko_button.clicked.connect(lambda: self.run_threaded_import(self.import_darko))
        button_layout.addWidget(darko_button)

        # Add your other buttons here...

        # Right side - Table view and progress display
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        splitter.addWidget(right_widget)

        # Setup table view with sorting enabled
        self.setup_table_view()
        right_layout.addWidget(self.table_view)


        # Add progress display
        self.progress_display = QTextEdit()
        self.progress_display.setReadOnly(True)
        self.progress_display.setMaximumHeight(100)
        right_layout.addWidget(self.progress_display)

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

    def import_bbm(self, progress_print=print):
        try:
            progress_print("Starting BBM import...")

            # Use the correct file path
            bbm_file = self.import_folder / "bbm.csv"

            if not bbm_file.exists():
                raise FileNotFoundError(f"bbm.csv file not found at {bbm_file}")

            progress_print(f"Reading file: {bbm_file}")
            df = pd.read_csv(bbm_file, index_col=False)
            progress_print(f"BBM data loaded successfully with {len(df)} rows")
            return df

        except Exception as e:
            progress_print(f"Error in BBM import: {str(e)}")
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

    def import_darko(self, progress_print=print):
        try:
            progress_print("Starting Darko import...")

            # Set up file paths
            darko_file = self.import_folder / "darko.csv"

            if not darko_file.exists():
                raise FileNotFoundError(f"darko.csv file not found at {darko_file}")

            # Read CSV data
            progress_print("Reading Darko CSV data...")
            try:
                # Define exact column names
                expected_columns = [
                    'nba_id',
                    'Team',
                    'Player',
                    'Experience',
                    'DPM',
                    'DPM Improvement',
                    'O-DPM',
                    'D-DPM',
                    'Box DPM',
                    'Box O-DPM',
                    'Box D-DPM',
                    'FGA/100',
                    'FG2%',
                    'FG3A/100',
                    'FG3%',
                    'FG3ARate%',
                    'RimFGA/100',
                    'RimFG%',
                    'FTA/100',
                    'FT%',
                    'FTARate%',
                    'USG%',
                    'REB/100',
                    'AST/100',
                    'AST%',
                    'BLK/100',
                    'BLK%',
                    'STL/100',
                    'STL%',
                    'TOV/100'
                ]

                data = pd.read_csv(darko_file)

                # Verify we have data
                if data.empty:
                    raise ValueError("No data was read from the Darko CSV file")

                progress_print(f"Successfully read {len(data)} rows of Darko data")

                # Verify columns match expected format
                if set(data.columns) != set(expected_columns):
                    progress_print("Warning: Columns don't match expected format")
                    progress_print(f"Expected columns: {expected_columns}")
                    progress_print(f"Found columns: {list(data.columns)}")

                def process_darko_data(df):
                    processed_df = df.copy()

                    # Convert numeric columns
                    percentage_columns = [
                        'FG2%', 'FG3%', 'FG3ARate%', 'RimFG%', 'FT%',
                        'FTARate%', 'USG%', 'AST%', 'BLK%', 'STL%'
                    ]

                    rate_columns = [
                        'FGA/100', 'FG3A/100', 'RimFGA/100', 'FTA/100',
                        'REB/100', 'AST/100', 'BLK/100', 'STL/100', 'TOV/100'
                    ]

                    dpm_columns = [
                        'DPM', 'DPM Improvement', 'O-DPM', 'D-DPM',
                        'Box DPM', 'Box O-DPM', 'Box D-DPM'
                    ]

                    # Convert percentages
                    for col in percentage_columns:
                        if col in processed_df.columns:
                            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

                    # Convert rate stats
                    for col in rate_columns:
                        if col in processed_df.columns:
                            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

                    # Convert DPM stats
                    for col in dpm_columns:
                        if col in processed_df.columns:
                            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

                    # Convert Experience to numeric
                    processed_df['Experience'] = pd.to_numeric(processed_df['Experience'], errors='coerce')

                    # Sort by DPM descending
                    processed_df = processed_df.sort_values('DPM', ascending=False)

                    return processed_df

                # Apply processing
                data = process_darko_data(data)

                # Store in dataframes dictionary
                self.dataframes['Darko'] = data

                # Update display if Darko is currently selected
                if self.data_selector.currentText() == 'Darko':
                    self.display_dataframe(data)

                progress_print("Darko import completed successfully")
                return data

            except pd.errors.EmptyDataError:
                raise ValueError("The Darko CSV file is empty")
            except Exception as e:
                raise ValueError(f"Error reading Darko CSV: {str(e)}")

        except Exception as e:
            progress_print(f"Error in Darko import: {str(e)}")
            raise



    def on_data_selection_changed(self, selection):
        if selection in self.dataframes:
            self.display_dataframe(self.dataframes[selection])

    def setup_table_view(self):
        self.table_view = QTableView()
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImportTool()
    window.show()
    sys.exit(app.exec())
