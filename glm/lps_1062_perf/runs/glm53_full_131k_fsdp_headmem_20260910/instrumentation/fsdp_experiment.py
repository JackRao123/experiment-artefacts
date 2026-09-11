"""Run-local, opt-in MFSDP wiring; single-adapter research only, not a config API."""
import logging
import time
from load_acceleration import accelerate_hf_import
import torch

logger = logging.getLogger(__name__)


def install():
    from trainers_server_megatron_bridge import chunked_lm_head
    chunked_lm_head.CHUNKED_LM_HEAD_SEQ_CHUNK = 2048
    logger.warning("EXPERIMENT: fused FP32 head operations, 2048-token head chunks")
    from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP, PrefetchOrder
    original_unshard = MegatronFSDP.all_gather_and_wait_parameters_ready
    def unshard(self, params, prefetch=True, prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER, wait_bucket_ready=True, bwd=False):
        return original_unshard(self, params, prefetch=False, prefetch_order=prefetch_order, wait_bucket_ready=wait_bucket_ready, bwd=bwd)
    MegatronFSDP.all_gather_and_wait_parameters_ready = unshard
    logger.warning("EXPERIMENT: parameter lookahead disabled; required all-gathers remain")
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV1
    from trainers_server_megatron_bridge import backend

    original_build = backend._build_config
    original_model = backend.get_model
    original_provider = backend.AutoBridge.to_megatron_provider

    def provider(self, load_weights=True, hf_path=None):
        # Loading the full model before wrapping would replicate all parameters
        # on every GPU. Existing Bridge's DTensor importer supports post-wrap load.
        if hf_path is not None:
            raise ValueError('FSDP probe uses base_model, not an override HF path')
        result = original_provider(self, load_weights=False)
        result._experiment_hf_bridge = self if load_weights else None
        return result

    def build(model_provider, config):
        if config.expert_weight_storage.value != 'bf16':
            raise ValueError('FSDP experiment currently requires BF16 storage')
        cfg = original_build(model_provider, config)
        cfg.dist.use_megatron_fsdp = True
        cfg.ddp.use_megatron_fsdp = True
        cfg.ddp.data_parallel_sharding_strategy = 'optim_grads_params'
        cfg.ddp.average_in_collective = False
        cfg.ddp.overlap_grad_reduce = True
        cfg.ddp.overlap_param_gather = True
        cfg.ddp.keep_fp8_transpose_cache = False
        cfg.ddp.fsdp_double_buffer = True
        cfg.ddp.megatron_fsdp_max_pool_double_buffer = True
        cfg.ddp.fsdp_db_use_persist_buf_on_alloc_fail = False
        cfg.model.init_model_with_meta_device = True
        cfg.checkpoint.ckpt_format = 'fsdp_dtensor'
        cfg.validate()
        logger.warning('EXPERIMENT: Megatron FSDP parameter sharding over DP+CP enabled')
        return cfg

    def get_model(model_provider, ddp_config, **kwargs):
        kwargs.update(use_megatron_fsdp=True, init_model_with_meta_device=True)
        started=time.perf_counter()
        models = original_model(model_provider, ddp_config, **kwargs)
        logger.warning('FSDP model construction completed in %.2f seconds; importing HF weights',time.perf_counter()-started)
        loaded_at=time.perf_counter()
        bridge = model_provider._experiment_hf_bridge
        if bridge is not None:
            with torch.no_grad(), accelerate_hf_import(bridge.hf_pretrained.model_name_or_path):
                bridge._model_bridge.load_weights_hf_to_megatron(bridge.hf_pretrained, models)
        logger.warning('Sharded HF import completed in %.2f seconds',time.perf_counter()-loaded_at)
        return models

    backend.AutoBridge.to_megatron_provider = provider
    backend._build_config = build
    backend.get_model = get_model
    # Both implementations provide zero_grad_buffer and scale_gradients. The
    # backend currently type-checks for DDP in those two places instead of the
    # shared interface. Keep this compatibility adaptation run-local first.
    backend.DistributedDataParallel = (backend.DistributedDataParallel, FullyShardedDataParallelV1)
