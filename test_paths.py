import os
import sys


def check_paths():
    print("\n=== Debug Path Info ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Executable path: {sys.executable}")
    print(f"Executable directory: {os.path.dirname(sys.executable)}")

    # Check for config.json
    config_path = os.path.join(os.path.dirname(sys.executable), 'config.json')
    print(f"\nChecking for config.json at: {config_path}")
    print(f"Exists: {os.path.exists(config_path)}")

    # Check for data directories
    print("\nChecking data directories:")
    for dir_name in ['dk_data', 'dk_import', 'dk_contests', 'dk_output']:
        dir_path = os.path.join(os.path.dirname(sys.executable), dir_name)
        print(f"{dir_name} path: {dir_path}")
        print(f"{dir_name} exists: {os.path.exists(dir_path)}")
        if os.path.exists(dir_path):
            print(f"{dir_name} contents: {os.listdir(dir_path)[:5]}")

    print("\nListing all files in executable directory:")
    exec_dir = os.path.dirname(sys.executable)
    print(os.listdir(exec_dir))

    # Also check _internal if it exists
    internal_path = os.path.join(os.path.dirname(sys.executable), '_internal')
    if os.path.exists(internal_path):
        print("\nContents of _internal directory:")
        print(os.listdir(internal_path))

    input("\nPress Enter to continue...")


if __name__ == "__main__":
    check_paths()