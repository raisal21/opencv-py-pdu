"""Expose the application entry point as `app.main()`."""
from importlib import import_module as _import_module


_main_mod = _import_module(".__main__", package=__name__)


main = _main_mod.main

__all__ = ["main"]
