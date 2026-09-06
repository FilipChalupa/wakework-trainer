import os
import sys
import tempfile
from pathlib import Path

# Point the app at a throw-away data directory before any app module is imported.
_TMP = Path(tempfile.mkdtemp(prefix="wakeword-test-"))
os.environ["DATA_DIR"] = str(_TMP)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
