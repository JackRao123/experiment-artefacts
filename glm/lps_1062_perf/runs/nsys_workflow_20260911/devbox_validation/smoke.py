"""Small standalone capability probe; not a trainer benchmark."""
import torch
import transformer_engine

print({"torch": torch.__version__, "transformer_engine": transformer_engine.__version__,
       "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()}, flush=True)
x = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    y = x @ x
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(1000):
    y = x @ x
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print({"finite": bool(y.isfinite().all()), "gpu": torch.cuda.get_device_name()}, flush=True)
