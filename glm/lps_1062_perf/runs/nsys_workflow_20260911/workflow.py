"""Single entry point: analyze TRACE; compare LEFT_JSON RIGHT_JSON; capture CASE; collect CASE."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
commands = {"analyze": "analyze.py", "compare": "compare.py", "capture": "capture.py", "collect": "collect.py"}
if len(sys.argv) < 2 or sys.argv[1] not in commands:
    raise SystemExit(__doc__)
raise SystemExit(subprocess.call([sys.executable, str(root / commands[sys.argv[1]]), *sys.argv[2:]]))
