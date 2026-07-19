"""Personal library for the adversarial-ember project (feature-space, phase 1)."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("adversarial-ember")
except PackageNotFoundError:  # imported from source without an install
    __version__ = "0+unknown"
