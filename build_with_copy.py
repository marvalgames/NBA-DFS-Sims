import os
import sys
import shutil
import subprocess
import pulp


def clean_dirs():
    """Clean build and dist directories"""
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}...")
            shutil.rmtree(dir_name)


def copy_files(dist_dir):
    """Copy files to their correct locations"""
    print("\nCopying files to distribution directory...")

    # Create necessary directories
    internal_dir = os.path.join(dist_dir, '_internal')
    os.makedirs(internal_dir, exist_ok=True)
    print(f"Created directory: {internal_dir}")

    # Copy config.json
    shutil.copy2('config.json', dist_dir)
    print(f"Copied config.json to {dist_dir}")

    # Copy source files to _internal
    src_files = ['nba_gpp_simulator.py', 'run_swap_sim.py', 'nba_swap_sims.py']
    for file in src_files:
        src_path = os.path.join('src', file)
        dst_path = os.path.join(internal_dir, file)  # Copy to _internal
        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Successfully copied {file} to {internal_dir}")
            else:
                print(f"WARNING: Source file not found: {src_path}")
        except Exception as e:
            print(f"Error copying {file}: {str(e)}")

    # Copy data directories
    for dir_name in ['dk_contests', 'dk_data', 'dk_output']:
        if os.path.exists(dir_name):
            dst_dir = os.path.join(dist_dir, dir_name)
            os.makedirs(dst_dir, exist_ok=True)
            for item in os.listdir(dir_name):
                s = os.path.join(dir_name, item)
                d = os.path.join(dst_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                    print(f"Copied {item} to {dst_dir}")

    # Copy PuLP solver
    solver = pulp.PULP_CBC_CMD(msg=False)
    if os.path.exists(solver.path):
        shutil.copy2(solver.path, dist_dir)
        print(f"Copied CBC solver to {dist_dir}")


def build_app():
    """Build the application"""
    print("Starting build process...")

    # Clean old builds
    clean_dirs()

    # Basic PyInstaller command
    cmd = [
        'pyinstaller',
        '--clean',
        '--name=NBA_GPP_Simulator',
        '--add-data', f'src/nba_gpp_simulator.py{os.pathsep}_internal',
        '--add-data', f'src/run_swap_sim.py{os.pathsep}_internal',
        '--add-data', f'src/nba_swap_sims.py{os.pathsep}_internal',
        '--hidden-import=pulp',
        '--hidden-import=pulp.apis',
        '--hidden-import=pulp.apis.core',
        '--hidden-import=pulp.apis.coin_api',
        '--hidden-import=pulp.solvers.coin',
        '--collect-submodules=pulp',
        '--paths=src',
        '--noconfirm',
        '--console',
        os.path.join('src', 'nba_sims_menu.py')
    ]

    print("Building executable...")
    subprocess.run(cmd, check=True)

    # Get the distribution directory
    dist_dir = os.path.join('dist', 'NBA_GPP_Simulator')

    # Copy additional files
    if os.path.exists(dist_dir):
        copy_files(dist_dir)
        print("\nBuild completed successfully!")
    else:
        print("\nError: Distribution directory not found!")


if __name__ == "__main__":
    try:
        build_app()
    except Exception as e:
        print(f"Error during build: {e}")