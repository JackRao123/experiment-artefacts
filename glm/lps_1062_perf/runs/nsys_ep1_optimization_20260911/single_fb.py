"""One untimed warmup, one timed/profiled FB, zero optimizer steps.

Copy beside capture.py on the devbox and run: python single_fb.py CASE
The one measured FB is profiled; there is no separate metrics/validation pass.
"""
from pathlib import Path
import runpy
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: python single_fb.py CASE")
runner = Path(__file__).resolve().parent / "capture.py"
sys.argv = [str(runner), sys.argv[1], "--fb-only", "--warmups", "1", "--controls", "0", "--trace-steps", "1"]
runpy.run_path(str(runner), run_name="__main__")
