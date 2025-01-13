# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import pulp
from PyInstaller.utils.hooks import collect_data_files

print("Starting build process...")

block_cipher = None

# Get absolute paths
root_dir = os.path.abspath(os.path.dirname(__file__))
print(f"Root directory: {root_dir}")

solver = pulp.PULP_CBC_CMD(msg=False)
cbc_path = solver.path
print(f"CBC solver path: {cbc_path}")

print("Setting up data files...")
# Define all data files
added_files = [
    # Config file at root level
    (os.path.join(root_dir, 'config.json'), '.'),
    # Data directories
    (os.path.join(root_dir, 'dk_contests'), 'dk_contests'),
    (os.path.join(root_dir, 'dk_data'), 'dk_data'),
    (os.path.join(root_dir, 'dk_output'), 'dk_output'),
    # Source files
    (os.path.join(root_dir, 'src', 'nba_gpp_simulator.py'), 'src'),
    (os.path.join(root_dir, 'src', 'run_swap_sim.py'), 'src'),
    (os.path.join(root_dir, 'src', 'nba_swap_sims.py'), 'src'),
    # PuLP solver
    (cbc_path, '.')
]

print("Configuring Analysis...")
a = Analysis(
    [os.path.join(root_dir, 'src', 'nba_sims_menu.py')],
    pathex=[os.path.join(root_dir, 'src')],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtWidgets',
        'PyQt6.QtGui',
        'pulp',
        'pulp.apis',
        'pulp.apis.core',
        'pulp.apis.coin_api',
        'pulp.solvers.coin',
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

print("Adding PuLP files...")
# Collect PuLP data files
pulp_datas = collect_data_files('pulp', include_py_files=True)
a.datas += pulp_datas

print("Creating PYZ archive...")
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

print("Creating EXE...")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NBA_GPP_Simulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

print("Creating COLLECT...")
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NBA_GPP_Simulator',
)

print("Build process completed!")