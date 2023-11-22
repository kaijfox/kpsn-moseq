
import pkgutil, importlib
from . import datasets
from . import transforms
from . import kpms_custom_io

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = importlib.import_module(f'{__name__}.{module_name}')
    globals()[module_name] = _module
