"""Shared pytest configuration for the project.

The repository root is put on the import path so that the ``pipeline``
package and the ``tools`` scripts can be imported by the tests without the
project having to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
