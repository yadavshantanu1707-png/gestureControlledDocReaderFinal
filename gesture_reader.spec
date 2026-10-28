# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Gesture Document Reader.
Bundles MediaPipe data files and hidden imports so the exe works standalone.
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

mediapipe_imports = collect_submodules('mediapipe')
mediapipe_datas  = collect_data_files('mediapipe')

a = Analysis(
    ['gesture_reader.py'],
    pathex=[],
    binaries=[],
    datas=mediapipe_datas,
    hiddenimports=[
        'cv2',
        'numpy',
        'google.protobuf',
        'google.protobuf.descriptor',
        'google.protobuf.descriptor_pool',
        'google.protobuf.runtime_version',
    ] + mediapipe_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GestureDocReader',
    debug=False,
    strip=False,
    upx=True,
    console=True,          # keep console for status messages
    icon=None,
    disable_windowed_traceback=False,
)
