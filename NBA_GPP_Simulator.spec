# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['pulp', 'pulp.apis', 'pulp.apis.core', 'pulp.apis.coin_api', 'pulp.solvers.coin']
hiddenimports += collect_submodules('pulp')


a = Analysis(
    ['src\\nba_sims_menu.py'],
    pathex=['src'],
    binaries=[],
    datas=[('src/nba_gpp_simulator.py', '_internal'), ('src/run_swap_sim.py', '_internal'), ('src/nba_swap_sims.py', '_internal')],
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
    name='NBA_GPP_Simulator',
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
    name='NBA_GPP_Simulator',
)
