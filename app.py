from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
PYTRAINER_DIR = BASE_DIR / "pytrainer"
if str(PYTRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(PYTRAINER_DIR))

from web_visualizer import app  # noqa: E402

