"""LPS-1003 instrument v2 loader: layer tracer + DSA top-k selection digest.
Each module is independently env-gated (BT_LTRACE=1 / BT_DSA_ROWSUM=1)."""
try:
    import ltrace_hooks  # noqa: F401
except Exception as e:  # noqa: BLE001
    import sys
    print(f"[harness2] ltrace_hooks failed: {e}", file=sys.stderr, flush=True)
try:
    import dsa_hooks  # noqa: F401
except Exception as e:  # noqa: BLE001
    import sys
    print(f"[harness2] dsa_hooks failed: {e}", file=sys.stderr, flush=True)
