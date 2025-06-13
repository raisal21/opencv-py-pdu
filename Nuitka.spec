# ===================================================================
#  pyside-deploy spec file for Eyelog
#  This conf is for a Windows standalone/onefile build using Nuitka.
# ===================================================================

[app]
# Title of your application, used for metadata.
title = Eyelog

# Project root directory. Default is '.', which is standard.
project_dir = .

# Source file entry point path, relative to project_dir.
input_file = run.py

# Directory where the executable output is generated.
exec_directory = release/nuitka

# Application icon, relative to project_dir.
icon = app/assets/icons/eyelog.ico

# Extra directories to explicitly ignore during the build process.
extra_ignore_dirs = app/tests,__pycache__

[python]
# Python packages required for the build.
# This tool can install them if they are not present.
packages = Nuitka==2.6.8

[qt]
# Qt modules used by the application, comma-separated.
modules = Core,OpenGLWidgets,Gui,Widgets,OpenGL,Charts

# Qt plugins used by the application.
# These are essential for desktop deployment.
plugins = platforms,styles,imageformats,iconengines,platforminputcontexts

[nuitka]
# The build mode. 'standalone' creates a distributable folder.
mode = standalone

# All extra Nuitka command-line arguments.
# This section is crucial and mirrors our successful command.
extra_args = --quiet ^
             --plugin-enable=pyside6 ^
             --plugin-enable=opencv-python ^
             --nofollow-import-to=*.tests ^
             --noinclude-qt-translations ^
             --include-data-dir=assets=assets ^
             --output-filename=EyeLog.exe
             --company-name="PT Parama Data Unit" ^
             --product-name="Eyelog" ^
             --file-version="1.0.0.0" ^
             --product-version="1.0.0.0" ^
             --copyright="© 2025 Parama Data Unit"