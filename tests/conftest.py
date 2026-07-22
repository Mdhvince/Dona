import sys
from pathlib import Path

# Modules in src/ use flat imports (from config import ...)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
