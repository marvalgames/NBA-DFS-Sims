# build_debug.py
import os
import pulp
import shutil
import subprocess


def print_structure():
    root = os.path.abspath(os.path.dirname(__file__))
    print("\nProject Structure:")
    print(f"Root: {root}")

    # Print PuLP solver info
    solver = pulp.PULP_CBC_CMD(msg=False)
    cbc_path = solver.path
    print(f"\nPuLP Solver:")
    print(f"CBC Path: {cbc_path}")
    print(f"Exists: {os.path.exists(cbc_path)}")

    # Print key file locations
    files_to_check = [
        'config.json',
        os.path.join('src', 'nba_sims_menu.py'),
        os.path.join('src', 'nba_gpp_simulator.py'),
        os.path.join('src', 'run_swap_sim.py'),
        os.path.join('src', 'nba_swap_sims.py'),
    ]

    print("\nKey Files:")
    for file in files_to_check:
        full_path = os.path.join(root, file)
        print(f"{file}: {'EXISTS' if os.path.exists(full_path) else 'MISSING'} at {full_path}")

    # Print directories
    dirs_to_check = ['dk_contests', 'dk_data', 'dk_output']
    print("\nDirectories:")
    for dir in dirs_to_check:
        full_path = os.path.join(root, dir)
        print(f"{dir}: {'EXISTS' if os.path.exists(full_path) else 'MISSING'} at {full_path}")


if __name__ == "__main__":
    print_structure()

    # Clean build directories
    dirs_to_clean = ['build', 'dist']
    for dir in dirs_to_clean:
        if os.path.exists(dir):
            print(f"\nCleaning {dir}...")
            shutil.rmtree(dir)

    # Build command with Windows path separators
    solver = pulp.PULP_CBC_CMD(msg=False)
    cbc_path = solver.path

    cmd = [
        'pyinstaller',
        '--clean',
        '--add-data', 'config.json;.',
        '--add-data', 'dk_contests;dk_contests',
        '--add-data', 'dk_data;dk_data',
        '--add-data', 'dk_output;dk_output',
        '--add-data', f'{cbc_path};.',
        '--add-data', 'src/nba_gpp_simulator.py;src',
        '--add-data', 'src/run_swap_sim.py;src',
        '--add-data', 'src/nba_swap_sims.py;src',
        '--hidden-import', 'pulp',
        '--hidden-import', 'pulp.apis',
        '--hidden-import', 'pulp.apis.core',
        '--hidden-import', 'pulp.apis.coin_api',
        '--hidden-import', 'pulp.solvers.coin',  # Changed from coin_cmd
        '--collect-submodules', 'pulp',  # Add this to collect all PuLP submodules
        'src/nba_sims_menu.py'
    ]

    print("\nExecuting build command...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("\nBuild Output:")
        print(result.stdout)

        if result.stderr:
            print("\nErrors:")
            print(result.stderr)

        if result.returncode == 0:
            print("\nBuild completed successfully!")
        else:
            print(f"\nBuild failed with return code: {result.returncode}")
    except Exception as e:
        print(f"\nError executing build command: {e}")