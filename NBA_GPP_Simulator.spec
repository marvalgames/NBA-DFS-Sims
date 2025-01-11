# NBA_GPP_Simulator.spec
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Print current directory for debugging
base_path = os.getcwd()
print(f"\nCurrent working directory: {base_path}")

# Define all required data files and folders
datas = [
    # Config file - copy to root of distribution
    ('config.json', '.'),
    # Data directories - maintain folder structure
    ('dk_data', 'dk_data'),
    ('dk_import', 'dk_import'),
    ('dk_contests', 'dk_contests'),
    ('dk_output', 'dk_output'),
]

# Add additional debug information
for src, dst in datas:
    full_path = os.path.join(base_path, src)
    if os.path.exists(full_path):
        if os.path.isfile(full_path):
            print(f"Found file: {src}")
        else:
            print(f"Found directory: {src} containing: {os.listdir(full_path)[:5]}...")
    else:
        print(f"WARNING: Cannot find {src} at {full_path}")

a = Analysis(
    [os.path.join('src', 'nba_sims_menu.py')],
    pathex=[base_path],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'numpy',
        'pandas',
        'csv',
        'json',
        'threading',
        'multiprocessing'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        # Add a runtime hook to set up correct paths
        'runtime_hook.py'
    ],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='NBA GPP Simulator',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    icon='icon.ico' if os.path.exists('icon.ico') else None
)