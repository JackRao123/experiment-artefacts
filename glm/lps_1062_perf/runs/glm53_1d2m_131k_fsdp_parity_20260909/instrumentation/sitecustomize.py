import os
import sys

is_trainer = any(arg in {"trainers_server_main.main", "trainers_server_interface.dp_worker.main"} for arg in sys.orig_argv)
if "RANK" in os.environ and os.environ.get("LAYER_TIMING_DIR") and is_trainer:
    try:
        from fsdp_experiment import install as install_fsdp
        if os.environ.get("PARITY_FSDP")=="1": install_fsdp()
        if os.environ.get("PARITY_GROUPED_MM")=="1":
            from frozen_grouped_mm import install as install_grouped
            install_grouped()
        from parity_init import install as install_parity
        install_parity()
        from layer_timing import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(91)
