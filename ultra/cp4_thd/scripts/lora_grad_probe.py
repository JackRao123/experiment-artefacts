"""Single-rank probe: do hybrid LoRA adapters receive gradients?

Boots MegatronBridgeController on the tiny NemotronH checkpoint (TP1/CP1),
runs one CE forward_backward, and prints per-adapter grad sums straight from
the parameter buffers. Run under torchrun inside the server venv:

    uv run --no-sync torchrun --nproc_per_node=1 lora_grad_probe.py \
        --model /root/.cache/user_artifacts/ultra_cp4/tiny_nemotron_h
"""

from __future__ import annotations

import argparse

import torch

from loops_models.control import RLControllerConfig
from loops_models.protocol import (
    Datum,
    ForwardBackwardDetails,
    ModelInput,
    TensorData,
)
from trainers_server.dp_worker.api.megatron_controller import (
    MegatronBridgeController,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq", type=int, default=192)
    args = parser.parse_args()

    cfg = RLControllerConfig.model_validate(
        {
            "base_model": args.model,
            "max_seq_len": 2048,
            "lora_rank": 16,
            "trust_remote_code": True,
            "weight_sync": {"type": "disabled"},
            "world_size": 1,
        }
    )
    controller = MegatronBridgeController(cfg)

    tokens = [(i % 613) + 17 for i in range(args.seq)]
    datum = Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=tokens[1:] + [-100], dtype="int64", shape=[args.seq]
            )
        },
    )
    result = controller.execute_forward_backward(
        ForwardBackwardDetails(data=[datum])
    )
    print(f"loss={result.loss:.6f}")

    trainable = 0
    nonzero_grads = 0
    zero_grads = []
    for name, p in controller._model_list[0].named_parameters():
        if not p.requires_grad:
            continue
        trainable += 1
        g = getattr(p, "main_grad", None)
        if g is None:
            g = p.grad
        gsum = None if g is None else float(g.abs().sum())
        if gsum and gsum > 0:
            nonzero_grads += 1
        else:
            zero_grads.append((name, gsum))
    print(f"trainable_params={trainable} nonzero_grad={nonzero_grads}")
    for name, gsum in zero_grads[:40]:
        print(f"ZERO {name} grad_sum={gsum}")


if __name__ == "__main__":
    main()
