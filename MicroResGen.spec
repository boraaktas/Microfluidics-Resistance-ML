# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

# Collecting libraries dynamically for dependencies like XGBoost and others
binaries = [
    ('C:/Users/vural/anaconda3/envs/conda_MFML/Library/bin/mkl_*.dll', '.'),
    ('C:/Users/vural/anaconda3/envs/conda_MFML/Library/bin/libiomp5md.dll', '.'),
    # Add glpsol executable
    ('C:/Users/vural/Desktop/Microfluidics-Resistance-ML/data/solver/glpsol.exe', '.'),
]

binaries += collect_dynamic_libs('xgboost')

# Hidden imports to resolve warnings
hiddenimports = []
hiddenimports += collect_submodules('pyomo')
hiddenimports += collect_submodules('sklearn')
hiddenimports += [
    'pkg_resources._vendor.jaraco.functools',
    'pkg_resources._vendor.jaraco.context',
    'pkg_resources._vendor.jaraco.text',
    'pkg_resources.extern',
    'scipy.special._cdflib',
    'joblib',
    'numpy.core._dtype_ctypes',
    'scipy._lib.messagestream',
]

# Adding data files for Matplotlib, Shapely, and XGBoost
datas = [
    ('C:/Users/vural/anaconda3/envs/conda_MFML/Lib/site-packages/matplotlib/mpl-data', 'matplotlib/mpl-data'),
    ('C:/Users/vural/anaconda3/envs/conda_MFML/Lib/site-packages/shapely.libs', 'shapely.libs'),
    ('C:/Users/vural/anaconda3/envs/conda_MFML/Lib/site-packages/xgboost/VERSION', 'xgboost'),
]

# Add logic for redirecting sys.stdout and sys.stderr to avoid 'flush' error
runtime_hook_code = """
import sys
import os
if not sys.stdout or sys.stdout.fileno() < 0:
    sys.stdout = open(os.devnull, 'w')
if not sys.stderr or sys.stderr.fileno() < 0:
    sys.stderr = open(os.devnull, 'w')
"""

# Write the runtime hook to a file
with open("runtime_hook.py", "w") as f:
    f.write(runtime_hook_code)

# Set up block cipher for secure packaging
a = Analysis(
    ['main_APP.py'],  # Replace with the entry-point script
    pathex=['.'],  # Add your source path here if needed
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],  # Add the runtime hook
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MicroResGen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Keep console hidden
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['data\\icon.ico'],  # Replace with your actual icon path
)
