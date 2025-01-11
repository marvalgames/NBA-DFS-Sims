import os
import sys


def _append_run_path():
    if getattr(sys, 'frozen', False):
        # Get the directory of the executable
        base_path = sys._MEIPASS

        # Ensure config.json is found in the root
        if not os.path.exists(os.path.join(base_path, 'config.json')):
            print(f"Warning: config.json not found in {base_path}")

        # Print debug info
        print(f"Runtime path setup:")
        print(f"Base path: {base_path}")
        print(f"Contents: {os.listdir(base_path)}")

        # Add base path to Python path
        if base_path not in sys.path:
            sys.path.insert(0, base_path)


_append_run_path()