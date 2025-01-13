import os
import sys
import shutil
import subprocess
import pulp


def clean_build_dirs():
    """Clean build and dist directories"""
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}...")
            shutil.rmtree(dir_name)


def build_app():
    """Build the application"""
    print("Starting build process...")

    # Get paths
    root_dir = os.path.abspath(os.path.dirname(__file__))
    solver = pulp.PULP_CBC_CMD(msg=False)
    cbc_path = solver.path

    print(f"Root directory: {root_dir}")
    print(f"CBC solver path: {cbc_path}")

    # Clean old builds
    clean_build_dirs()

    # Define paths with proper Windows separators
    config_path = os.path.join(root_dir, 'config.json')
    contests_path = os.path.join(root_dir, 'dk_contests')
    data_path = os.path.join(root_dir, 'dk_data')
    output_path = os.path.join(root_dir, 'dk_output')
    src_path = os.path.join(root_dir, 'src')

    # Construct PyInstaller command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        '-m',
        'PyInstaller',
        '--clean',
        '--name=NBA_GPP_Simulator',
        f'--add-data={config_path};.',
        f'--add-data={contests_path};dk_contests',
        f'--add-data={data_path};dk_data',
        f'--add-data={output_path};dk_output',
        f'--add-data={cbc_path};.',
        f'--add-data={os.path.join(src_path, "nba_gpp_simulator.py")};src',
        f'--add-data={os.path.join(src_path, "run_swap_sim.py")};src',
        f'--add-data={os.path.join(src_path, "nba_swap_sims.py")};src',
        '--hidden-import=pulp',
        '--hidden-import=pulp.apis',
        '--hidden-import=pulp.apis.core',
        '--hidden-import=pulp.apis.coin_api',
        '--hidden-import=pulp.solvers.coin',
        '--collect-submodules=pulp',
        f'--paths={src_path}',
        '--noconfirm',
        '--console',
        os.path.join(src_path, 'nba_sims_menu.py')
    ]

    print("\nExecuting command:")
    print(' '.join(cmd))

    # Run PyInstaller
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # Print output in real-time
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()

            if output:
                print(output.strip())
            if error:
                print(error.strip(), file=sys.stderr)

            # Check if process has finished
            if output == '' and error == '' and process.poll() is not None:
                break

        if process.returncode == 0:
            print("\nBuild completed successfully!")

            # Verify dist folder contents
            dist_path = os.path.join(root_dir, 'dist', 'NBA_GPP_Simulator')
            if os.path.exists(dist_path):
                print("\nDist folder contents:")
                for root, dirs, files in os.walk(dist_path):
                    level = root.replace(dist_path, '').count(os.sep)
                    indent = ' ' * 4 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for f in files:
                        print(f"{subindent}{f}")
            else:
                print("\nWarning: Dist folder not found!")
        else:
            print(f"\nBuild failed with return code: {process.returncode}")

    except Exception as e:
        print(f"Error during build: {e}")


if __name__ == "__main__":
    build_app()