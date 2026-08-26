from types import SimpleNamespace

import torch

from deep_ep import Buffer, HybridEPBuffer
from loops_models.control import TrainerControllerConfig
from trainers_server_interface.dp_worker.gpu_name_shim import apply
from trainers_server_megatron_bridge.megatron_config import _configure_moe_provider


def validate_backend(backend: str, num_sms: int) -> None:
    config = TrainerControllerConfig.model_validate(
        {
            "base_model": "/root/.cache/user_artifacts/glm52-debug-1d1m",
            "max_seq_len": 8192,
            "weight_sync": {"type": "disabled"},
            "moe_token_dispatcher": "flex",
            "moe_flex_dispatcher_backend": backend,
            "moe_flex_dispatcher_num_sms": num_sms,
        }
    )
    provider = SimpleNamespace(
        moe_token_dispatcher_type="alltoall",
        moe_flex_dispatcher_backend="deepep",
        moe_flex_dispatcher_num_sms=20,
        moe_shared_expert_overlap=True,
    )
    _configure_moe_provider(provider, config)
    assert provider.moe_token_dispatcher_type == "flex"
    assert provider.moe_flex_dispatcher_backend == backend
    assert provider.moe_flex_dispatcher_num_sms == num_sms
    assert provider.moe_shared_expert_overlap is False


apply()
assert torch.cuda.get_device_name().startswith("NVIDIA B300")
assert Buffer is not None
assert HybridEPBuffer is not None
validate_backend("deepep", 20)
validate_backend("hybridep", 16)
print("flex config validation passed")
