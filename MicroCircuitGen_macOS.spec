# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs
from PyInstaller.utils.hooks import collect_data_files

# Paths
CONDA_ENV_PATH = '/opt/anaconda3/envs/Microfluidics-Resistance-ML'
PROJECT_ROOT = '/Users/bora/Desktop/DXBiotechLab/Microfluidics-Resistance-ML'

# Collect dynamic libraries for xgboost
binaries = collect_dynamic_libs('xgboost')

# Manually add libxgboost.dylib
binaries.append((
    os.path.join(CONDA_ENV_PATH, 'lib/python3.11/site-packages/xgboost/lib/libxgboost.dylib'),
    'xgboost/lib'
))
print('---------------------------binaries---------------------------')
print(binaries)

# Hidden imports
hiddenimports = []
hiddenimports += collect_submodules('pyomo')
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('xgboost')
hiddenimports += collect_submodules('shapely')
hiddenimports += collect_submodules('joblib')
hiddenimports += collect_submodules('scipy.special')
hiddenimports += collect_submodules('numpy.core')
hiddenimports += collect_submodules('scipy._lib')
print('---------------------------hiddenimports---------------------------')
print(hiddenimports)

# Collect data files
datas = []
datas += collect_data_files('matplotlib', include_py_files=True)
datas += collect_data_files('shapely', include_py_files=False)
datas.append((os.path.join(CONDA_ENV_PATH, 'lib/python3.11/site-packages/xgboost/VERSION'), 'xgboost'))
datas += [
    (os.path.join(PROJECT_ROOT, 'data/'), 'data/'),
]
print('---------------------------datas---------------------------')
print(datas)

# Solver binary (glpsol)

# Runtime hook
runtime_hook_code = """
import sys
import os
if not sys.stdout or sys.stdout.fileno() < 0:
    sys.stdout = open(os.devnull, 'w')
if not sys.stderr or sys.stderr.fileno() < 0:
    sys.stderr = open(os.devnull, 'w')
"""

with open("runtime_hook.py", "w") as f:
    f.write(runtime_hook_code)

block_cipher = None

# PyInstaller Analysis
a = Analysis(
    ['main_APP.py'],  # Entry point
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MicroCircuitGen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # it is not suggested for macOS
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Keep console hidden
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[os.path.join(PROJECT_ROOT, 'data/icon.ico')]  # macOS-compatible icon
)
