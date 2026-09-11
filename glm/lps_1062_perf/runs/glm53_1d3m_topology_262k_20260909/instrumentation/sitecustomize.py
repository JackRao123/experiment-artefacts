"""Run-only instrumentation, loaded only in distributed workers."""
import os
if "RANK" in os.environ and os.environ.get("LAYER_TIMING_DIR"):
    try:
        from layer_timing import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(91)
