# The package __init__.py uses relative imports that only resolve under
# ComfyUI's loader, so keep pytest from importing it as the root package.
# Tests load the modules themselves with stub parent packages.
from _pytest.python import Package

_orig_setup = Package.setup


def _setup(self):
    if self.path == self.config.rootpath:
        return
    _orig_setup(self)


Package.setup = _setup
collect_ignore = ["__init__.py"]
