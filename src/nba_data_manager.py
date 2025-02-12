from datetime import datetime, timezone, timedelta

import pandas as pd


class NBADataManager:
    def __init__(self):
        self.data = {}  # Dictionary to store different DataFrames
        self.current_date = datetime.now(timezone(timedelta(hours=-5))).date()

    def load_csv(self, file_path, key):
        """Load CSV file into DataFrame storage"""
        try:
            self.data[key] = pd.read_csv(file_path)
            return True
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False

    def save_csv(self, key, file_path):
        """Save DataFrame to CSV"""
        try:
            if key in self.data:
                self.data[key].to_csv(file_path, index=False)
                return True
            return False
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            return False

    def get_data(self, key):
        """Get DataFrame by key"""
        return self.data.get(key)

    def set_data(self, key, df):
        """Set DataFrame by key"""
        self.data[key] = df