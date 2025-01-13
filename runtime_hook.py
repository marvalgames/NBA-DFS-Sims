# # runtime_hook.py
# import os
# import sys
#
#
# def ensure_directories():
#     """Ensure required directories exist."""
#     if getattr(sys, 'frozen', False):
#         base_dir = sys._MEIPASS
#     else:
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#
#     directories = ['dk_output', 'dk_contests', 'dk_data']
#     for dir_name in directories:
#         dir_path = os.path.join(base_dir, dir_name)
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#
#
# # This will run when the application starts
# ensure_directories()