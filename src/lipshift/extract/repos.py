"""Import isolation for the two upstream VSR code bases.

Auto-AVSR and the multilingual VSR repository both ship top-level packages named
``espnet`` (and other shared names) with different contents. ``switch_repo`` purges the
cached modules and puts the requested repository first on ``sys.path``. Encoders that
were already built keep working, because their classes stay referenced.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_PACKAGES = ("espnet", "datamodule", "preparation", "pipelines")


def switch_repo(repo_dir: str | Path, packages: tuple[str, ...] = REPO_PACKAGES) -> None:
    """Make ``repo_dir`` the source of ``packages`` for subsequent imports."""
    for name in list(sys.modules):
        if name.split(".")[0] in packages:
            del sys.modules[name]
    p = str(Path(repo_dir).resolve())
    sys.path[:] = [x for x in sys.path if x != p]
    sys.path.insert(0, p)
