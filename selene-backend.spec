# -*- mode: python ; coding: utf-8 -*-

import sys

from PyInstaller.utils.hooks import collect_all

block_cipher = None

chromadb_datas, chromadb_binaries, chromadb_hiddenimports = collect_all(
    'chromadb',
    filter_submodules=lambda name: not name.startswith('chromadb.test'),
)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=chromadb_binaries,
    datas=[
        ('Modelfile', '.'),
        ('agent/prompts/external_models.md', 'agent/prompts'),
        ('agent/static', 'agent/static'),
        *chromadb_datas,
    ],
    hiddenimports=[
        'ollama', 'rich', 'requests', 'pdf2image', 'pypdf', 'docx', 'ddgs',
        *chromadb_hiddenimports,
        *(['dbus'] if sys.platform.startswith('linux') else []),
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='selene-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    # Keep stdout available for the readiness/port contract. Electron spawns
    # this backend with windowsHide=true, so packaged Windows users do not see
    # a stray console while startup diagnostics remain capturable.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
