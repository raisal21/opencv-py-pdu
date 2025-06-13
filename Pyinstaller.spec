# -*- mode: python ; coding: utf-8 -*-

# This is a PyInstaller spec file. It is a Python script that PyInstaller
# uses to package your application.

# To build the application, run in your terminal:
# pyinstaller your_spec_file.spec

# Using a block cipher for PYZ is optional. None is standard.
block_cipher = None

# --- Analysis: Finding all the necessary files ---
# This is the main step where PyInstaller analyzes your code to find
# all dependencies and modules.
a = Analysis(
    ['run.py'],  # The main entry point of your application
    pathex=[],
    binaries=[],

    # --- Data Files ---
    # This tells PyInstaller where to find non-code assets (images, icons, etc.)
    # Format: [ ('source_path_on_disk', 'destination_in_dist_folder') ]
    # This copies the entire 'assets' folder into the root of the dist folder.
    datas=[('assets', 'assets')],

    # --- Hidden Imports ---
    # Modules that are not automatically detected by PyInstaller.
    # PySide6 hooks usually handle most things, but some can be missed.
    # Add any module here that causes an 'ImportError' at runtime.
    hiddenimports=[
        'PySide6.QtCharts',
        'PySide6.QtOpenGL',
        # Add other specific modules if needed, e.g., for opencv
        # 'cv2',
    ],
    
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # --- Excludes ---
    # Modules to explicitly exclude from the build to save space.
    excludes=['tests', 'pytest', 'unittest'],

    # --- Options ---
    noarchive=False,
    optimize=0,
)

# --- PYZ: Creating the Python library archive ---
# This bundles all the Python modules found in the Analysis step
# into a single compressed archive.
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- EXE: Building the main executable ---
# This creates the .exe file itself, linking it to the Python archive.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,

    # --- Executable Metadata ---
    name='EyeLog', # The name of your .exe file
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,

    # --- Console Window Control ---
    # 'False' for GUI applications to hide the black console window.
    # This is the equivalent of Nuitka's --windows-disable-console.
    console=False,
    
    # --- Application Icon ---
    # Path to your application's .ico file.
    icon='assets/icons/eyelog.ico',
    
    # You can also embed version information from a file
    # version='version.txt',
)

# --- COLLECT: Assembling the final distribution folder ---
# This step gathers the EXE, the PYZ archive, data files, and all required
# DLLs into a single output folder. This is the 'standalone' part.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='EyeLog',
)