"""Tasks of muscle_supermimic, registered with mjlab's task registry on import.

The ``mjlab.tasks`` entry point in pyproject.toml makes mjlab import this package, so mjlab's
``train`` / ``play`` CLIs list the tasks registered under ``*/config/*``.
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp", ".scripts"]

import_packages(__name__, _BLACKLIST_PKGS)
