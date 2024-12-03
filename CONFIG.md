    "player_path": "DKSalaries.csv",
    "projection_path": "Projections.csv",
    "late_swap_path": "DKEntries.csv",


    "projection_path": "ProjectionsLate.csv",
    "player_path": "DKSalariesLate.csv",
    "late_swap_path": "DKEntriesLate.csv",


pyinstaller --add-binary="D:\BLBRAMOS\PyVirtualAcer\.venv\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py
pyinstaller --add-binary="C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py

pyinstaller --add-binary="C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py

pyinstaller --add-data "run_swap_sim.py;." --add-binary "C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py
