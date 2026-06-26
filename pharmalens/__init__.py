"""PharmaLens pharmaceutical document knowledge base."""

from pathlib import Path

from dotenv import load_dotenv

_package_root = Path(__file__).resolve().parent
load_dotenv(_package_root.parent / ".env")
load_dotenv(_package_root / ".env")

__version__ = "0.1.0"
