#!/usr/bin/env python3
"""Run the shared profile driver with output on node-local scratch."""

from pathlib import Path

import profile_driver


profile_driver.OUT_DIR = Path("/tmp/lps1062_native_blockwise_bench")
profile_driver.main()
