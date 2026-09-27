import sys
from pathlib import Path

# The engine modules import each other by top-level name (``from
# core.portfolio import Side``) while the tests import them as ``src.*``.
# Both roots have to be importable for the two styles to meet.
#
# Consequence to keep in mind: ``src.orders.models`` and ``orders.models``
# end up as two *separate* module objects, so their enum members are equal
# but not identical. Compare enum members with ``==``, never with ``is``.
ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
