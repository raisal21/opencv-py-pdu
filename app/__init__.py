"""
app/__init__.py
Mengekspor fungsi main() dari app.__main__ supaya dapat dipanggil
melalui `import app; app.main()`.
"""
from importlib import import_module as _import_module

# Import modul eksekusi di dalam paket
_main_mod = _import_module(".__main__", package=__name__)

# Re-export fungsi main()
main = _main_mod.main

__all__ = ["main"]       # opsional, membantu linters & autocomplete
