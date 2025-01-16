import os
import sys
import shutil
import subprocess
import pulp
import filecmp


def clean_dirs():
    """Clean build and dist directories"""
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}...")
            shutil.rmtree(dir_name)


def copy_executable(app_name):
    """Copy executable and its dependencies from subdirectory to main directory"""
    source_dir = os.path.join('dist', 'NBA_Tools', app_name)
    target_dir = os.path.join('dist', 'NBA_Tools')

    # Copy the executable
    exe_name = f"{app_name}.exe"
    source_exe = os.path.join(source_dir, exe_name)
    target_exe = os.path.join(target_dir, exe_name)

    if os.path.exists(source_exe):
        shutil.copy2(source_exe, target_exe)
        print(f"Copied {exe_name} to main directory")

    # Copy all contents from source _internal to shared _internal
    source_internal = os.path.join(source_dir, '_internal')
    target_internal = os.path.join(target_dir, '_internal')

    if os.path.exists(source_internal):
        # Copy all files and subdirectories from source _internal
        for item in os.listdir(source_internal):
            s = os.path.join(source_internal, item)
            d = os.path.join(target_internal, item)

            if os.path.isfile(s):
                # Copy file if it doesn't exist or is different
                if not os.path.exists(d) or not filecmp.cmp(s, d, shallow=False):
                    shutil.copy2(s, d)
                    print(f"Copied dependency {item} to shared _internal")
            elif os.path.isdir(s):
                # Copy directory and its contents
                if not os.path.exists(d):
                    shutil.copytree(s, d)
                    print(f"Copied dependency directory {item} to shared _internal")
                else:
                    # Merge directory contents
                    for root, dirs, files in os.walk(s):
                        rel_path = os.path.relpath(root, s)
                        target_root = os.path.join(d, rel_path)
                        os.makedirs(target_root, exist_ok=True)
                        for file in files:
                            src_file = os.path.join(root, file)
                            dst_file = os.path.join(target_root, file)
                            if not os.path.exists(dst_file) or not filecmp.cmp(src_file, dst_file, shallow=False):
                                shutil.copy2(src_file, dst_file)
                                print(f"Copied dependency {os.path.join(rel_path, file)} to shared _internal")


def setup_shared_directory():
    """Create and setup the shared directory structure"""
    dist_dir = os.path.join('dist', 'NBA_Tools')
    os.makedirs(dist_dir, exist_ok=True)

    # Create _internal directory
    internal_dir = os.path.join(dist_dir, '_internal')
    os.makedirs(internal_dir, exist_ok=True)

    # Copy config.json
    if os.path.exists('config.json'):
        shutil.copy2('config.json', dist_dir)
        print(f"Copied config.json to {dist_dir}")

    # Copy source files to _internal
    src_files = [
        'nba_gpp_simulator.py',
        'run_swap_sim.py',
        'nba_swap_sims.py',
        'nba_importer_menu.py'
    ]

    for file in src_files:
        src_path = os.path.join('src', file)
        dst_path = os.path.join(internal_dir, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {file} to {internal_dir}")

    # Copy data directories
    for dir_name in ['dk_contests', 'dk_data', 'dk_output', 'dk_import']:
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


def build_app(app_name, main_script, additional_files=None):
    """Build a single executable"""
    print(f"Building {app_name}...")

    work_dir = os.path.join('build', app_name)
    os.makedirs(work_dir, exist_ok=True)

    cmd = [
        'pyinstaller',
        '--clean',
        f'--name={app_name}',
        f'--workpath={work_dir}',
        '--distpath=dist/NBA_Tools'
    ]

    # Add data files
    if additional_files:
        for file in additional_files:
            cmd.extend(['--add-data', f'src/{file}{os.pathsep}_internal'])

    # Add common options
    cmd.extend([
        '--hidden-import=pulp',
        '--hidden-import=pulp.apis',
        '--hidden-import=pulp.apis.core',
        '--hidden-import=pulp.apis.coin_api',
        '--hidden-import=pulp.solvers.coin',
        '--collect-submodules=pulp',
        '--paths=src',
        '--noconfirm',
        '--console',
        os.path.join('src', main_script)
    ])

    print("Building executable...")
    subprocess.run(cmd, check=True)

    # Move executable to main directory and cleanup
    copy_executable(app_name)


if __name__ == "__main__":
    try:
        # Clean old builds
        clean_dirs()

        # Setup shared directory structure first
        setup_shared_directory()

        # Build both executables
        simulator_files = ['nba_gpp_simulator.py', 'run_swap_sim.py', 'nba_swap_sims.py']
        build_app('NBA_GPP_Simulator', 'nba_sims_menu.py', simulator_files)
        build_app('NBA_Importer', 'nba_importer_menu.py')

        print("\nBuild completed successfully!")
        print("Both executables and shared data are in the 'dist/NBA_Tools' directory")

    except Exception as e:
        print(f"Error during build: {e}")