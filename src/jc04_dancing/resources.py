"""Locate installed assets independently of the working directory."""
from pathlib import Path


def resource_path(relative: str) -> Path:
    """Return an existing package path; reject requests outside the package."""
    root = Path(__file__).resolve().parent
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Resource must be relative to the package: {relative}")
    if not path.exists():
        raise FileNotFoundError(f"Package resource not found: {relative}")
    return path
