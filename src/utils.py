# src/utils.py
import os
import sys


def get_project_root():
    """Get the project root directory"""
    if getattr(sys, '_MEIPASS', False):
        return sys._MEIPASS

    # When running from src directory, go up one level
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)  # Go up one directory from src


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = get_project_root()
    return os.path.join(base_path, relative_path)


def get_output_path(filename):
    """Get the correct path for output files."""
    output_dir = os.path.join(get_project_root(), 'dk_output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return os.path.join(output_dir, filename)