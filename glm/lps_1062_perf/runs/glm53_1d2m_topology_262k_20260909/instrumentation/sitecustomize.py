"""Run-only instrumentation, loaded only in distributed workers."""
import os
import sys
is_compiler = any("compile_worker" in arg for arg in sys.orig_argv)
if "RANK" in os.environ and os.environ.get("LAYER_TIMING_DIR") and not is_compiler:
    try:
        from layer_timing import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(91)
