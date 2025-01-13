# -*- mode: python ; coding: utf-8 -*-
import os
import pulp

block_cipher = None

# Get absolute paths
root_dir = os.path.abspath(os.path.dirname(__file__))
solver = pulp.PULP_CBC_CMD(msg=False)
cbc_path = solver.path

# Define all file locations
added_files = [
    # Config at root level
    (os.path.join(root_dir, 'config.json'), '.'),

    # Data directories with their structure
    (os.path.join(root_dir, 'dk_contests'), 'dk_contests'),
    (os.path.join(root_dir, 'dk_data'), 'dk_data'),
    (os.path.join(root_dir, 'dk_output'), 'dk_output'),

    # Source files in src directory
    (os.path.join(root_dir, 'src', 'nba_gpp_simulator.py'), 'src'),
    (os.path.join(root_dir, 'src', 'run_swap_sim.py'), 'src'),
    (os.path.join(root_dir, 'src', 'nba_swap_sims.py'), 'src'),

    # PuLP solver
    (cbc_path, '.')
]

# Print file locations for verification
print("Files to be included:")
for src, dst in added_files:
    print(f"Source: {src} -> Destination: {dst}")

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

# Include all PuLP-related files
pulp_dir = os.path.dirname(pulp.__file__)
a.datas += [(os.path.join('pulp', os.path.relpath(os.path.join(root, file), pulp_dir)), os.path.join(root, file), 'DATA')
            for root, dirs, files in os.walk(pulp_dir)
            for file in files]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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