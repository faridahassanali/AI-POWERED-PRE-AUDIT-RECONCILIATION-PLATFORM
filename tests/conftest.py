import sys
from pathlib import Path


# Get the project root:
# AI-POWERED-PRE-AUDIT-RECONCILIATION-PLATFORM
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Add the project root to Python's import path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))