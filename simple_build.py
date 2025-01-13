import os
import subprocess
import shutil

# Clean old builds
for dir_name in ['build', 'dist']:
    if os.path.exists(dir_name):
        print(f"Cleaning {dir_name}...")
        shutil.rmtree(dir_name)

# Basic command
cmd = [
    'pyinstaller',
    '--clean',
    '--name=NBA_GPP_Simulator',
    '--add-data=config.json;.',
    '--add-data=dk_contests;dk_contests',
    '--add-data=dk_data;dk_data',
    '--add-data=dk_output;dk_output',
    '--add-data=src/nba_gpp_simulator.py;src',
    '--add-data=src/run_swap_sim.py;src',
    '--add-data=src/nba_swap_sims.py;src',
    '--paths=src',
    '--console',
    'src/nba_sims_menu.py'
]

# Execute command
result = subprocess.run(cmd, capture_output=True, text=True)

# Print output
print("\nOutput:")
print(result.stdout)

if result.stderr:
    print("\nErrors:")
    print(result.stderr)

print("\nChecking dist folder...")
dist_path = 'dist/NBA_GPP_Simulator'
if os.path.exists(dist_path):
    print("\nDist folder contents:")
    for root, dirs, files in os.walk(dist_path):
        for f in files:
            print(f"  {f}")
else:
    print("Dist folder not found!")