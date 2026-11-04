import sys
from pathlib import Path

# Run from a source checkout without installing the adapter.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
