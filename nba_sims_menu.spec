# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['pulp', 'pulp.apis', 'pulp.apis.core', 'pulp.apis.coin_api', 'pulp.solvers.coin']
hiddenimports += collect_submodules('pulp')


a = Analysis(
    ['src\\nba_sims_menu.py'],
    pathex=[],
    binaries=[],
    datas=[('config.json', '.'), ('dk_contests', 'dk_contests'), ('dk_data', 'dk_data'), ('dk_output', 'dk_output'), ('C:\\Python\\pyVirtual\\.venv\\Lib\\site-packages\\pulp\\solverdir\\cbc\\win\\64\\cbc.exe', '.'), ('src/nba_gpp_simulator.py', 'src'), ('src/run_swap_sim.py', 'src'), ('src/nba_swap_sims.py', 'src')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='nba_sims_menu',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='nba_sims_menu',
)
