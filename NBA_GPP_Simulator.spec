# NBA_GPP_Simulator.spec
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

a = Analysis(
    [os.path.join('src', 'nba_sims_menu.py')],
    pathex=[],
    binaries=[],
    datas=[],  # Empty datas - we'll handle in COLLECT
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
    runtime_hooks=[],
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
    [],  # Empty list for directory build
    exclude_binaries=True,  # Important for directory build
    name='NBA GPP Simulator',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)

# Collect all files
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    [
        ('config.json', 'config.json', 'DATA'),
        ('dk_data', 'dk_data', 'DATA'),
        ('dk_import', 'dk_import', 'DATA'),
        ('dk_contests', 'dk_contests', 'DATA'),
        ('dk_output', 'dk_output', 'DATA'),
    ],
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NBA GPP Simulator',
)