    "player_path": "DKSalaries.csv",
    "projection_path": "Projections.csv",
    "late_swap_path": "DKEntries.csv",


    "projection_path": "ProjectionsLate.csv",
    "player_path": "DKSalariesLate.csv",
    "late_swap_path": "DKEntriesLate.csv",


Python
-------------------------------------------
C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Scripts 

pyinstaller --add-binary="D:\BLBRAMOS\PyVirtualAcer\.venv\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py
pyinstaller --add-binary="C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py

pyinstaller --add-binary="C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py

pyinstaller --add-data "run_swap_sim.py;." --add-binary "C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py
pyinstaller --add-data "run_swap_sim.py;_internal" --add-binary "C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py


G:\My Drive\NBA-DFS-Tools\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe
pyinstaller --hidden-import=requests --add-data "run_swap_sim.py;." --add-data "nba_swap_sims.py;." --add-binary "G:\My Drive\NBA-DFS-Tools\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py

pyinstaller --hidden-import=requests --add-data "run_swap_sim.py;." --add-data "nba_swap_sims.py;."  --add-binary "C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Scripts\python.exe;."  --add-binary "C:\Users\Ramos\My Drive (blbramos@gmail.com)\PyVirtual\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." main_app.py
cd "C:\Users\Ramos\My Drive (blbramos@gmail.com)\STATS\DFS 2024\NBA-DFS-Tools\src"
pyinstaller --hidden-import=requests --hidden-import=pulp --add-data "run_swap_sim.py;." --add-data "nba_swap_sims.py;."  --add-binary ""C:\Python\pyVirtual\.venv\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe";." main_app.py

pyinstaller --hidden-import=requests --hidden-import=pulp --add-data "run_swap_sim.py;."  --add-data "nba_swap_sims.py;."  --add-binary "C:\Python\pyVirtual\.venv\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe;." --distpath "C:\Python\dfs\dist"  --workpath "C:\Python\dfs\build"  main_app.py

