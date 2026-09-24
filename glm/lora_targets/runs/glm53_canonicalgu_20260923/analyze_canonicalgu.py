"""Inspect saved LoRA factors on CPU without loading the base model."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def inspect(path: Path) -> dict:
    rows = []
    counts = Counter()
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        for key in keys:
            group = (
                "routed" if ".experts." in key else
                "shared" if ".shared_experts." in key else
                "dense" if ".mlp." in key else
                "attention" if ".self_attn." in key else "other"
            )
            counts[group] += math.prod(tensors.get_slice(key).get_shape())
        for key in keys:
            if not key.endswith("gate_proj.lora_A.weight") or ".experts." in key:
                continue
            ag = tensors.get_tensor(key).double()
            au = tensors.get_tensor(key.replace("gate_proj", "up_proj")).double()
            bg = tensors.get_tensor(key.replace("lora_A", "lora_B")).double()
            bu = tensors.get_tensor(key.replace("gate_proj", "up_proj").replace("lora_A", "lora_B")).double()
            stacked_a = torch.cat([ag, au])
            _, r = torch.linalg.qr(stacked_a.T, mode="reduced")
            gram_b = torch.block_diag(bg.T @ bg, bu.T @ bu)
            eigenvalues = torch.linalg.eigvalsh(r @ gram_b @ r.T).clamp_min(0).flip(0)
            total_energy = eigenvalues.sum().item()
            gate_energy = ((bg.T @ bg) * (ag @ ag.T)).sum().item()
            up_energy = ((bu.T @ bu) * (au @ au.T)).sum().item()
            rows.append({
                "module": key.removesuffix(".gate_proj.lora_A.weight"),
                "a_equal": torch.equal(ag, au),
                "gate_b_norm": bg.norm().item(),
                "up_b_norm": bu.norm().item(),
                "gate_delta_norm": math.sqrt(max(gate_energy, 0)),
                "up_delta_norm": math.sqrt(max(up_energy, 0)),
                "rank32_residual_energy_fraction": eigenvalues[32:].sum().item() / total_energy if total_energy else None,
                "rank_for_90pct_energy": int((eigenvalues.cumsum(0) < 0.9 * total_energy).sum().item()) + 1,
                "finite": bool(all(torch.isfinite(t).all() for t in [ag, au, bg, bu])),
            })
        routed_gate_a = [k for k in keys if k.endswith(".experts.gate_proj.lora_A.weight")]
        routed_equal = sum(torch.equal(tensors.get_tensor(k), tensors.get_tensor(k.replace("gate_proj", "up_proj"))) for k in routed_gate_a)
        duplicate_routed_a = sum(math.prod(tensors.get_slice(k).get_shape()) for k in routed_gate_a)
        duplicate_dense_shared_a = sum(
            math.prod(tensors.get_slice(row["module"] + ".gate_proj.lora_A.weight").get_shape())
            for row in rows if row["a_equal"]
        )
    return {
        "path": str(path),
        "exported_parameter_counts": dict(counts),
        "unique_parameter_count_accounting_for_gate_up_tying": sum(counts.values()) - duplicate_routed_a - duplicate_dense_shared_a,
        "dense_shared_modules": len(rows),
        "equal_gate_up_a": sum(row["a_equal"] for row in rows),
        "nonzero_gate_b": sum(row["gate_b_norm"] > 0 for row in rows),
        "nonzero_up_b": sum(row["up_b_norm"] > 0 for row in rows),
        "all_finite": all(row["finite"] for row in rows),
        "routed_shared_a_count": len(routed_gate_a),
        "routed_equal_gate_up_a": routed_equal,
        "rank32_residual_energy_mean": statistics.mean(row["rank32_residual_energy_fraction"] for row in rows),
        "rank32_residual_energy_median": statistics.median(row["rank32_residual_energy_fraction"] for row in rows),
        "rank90_median": statistics.median(row["rank_for_90pct_energy"] for row in rows),
        "modules": rows,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--canonical", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
result = {"baseline": inspect(args.baseline), "canonical": inspect(args.canonical)}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: {field: value for field, value in v.items() if field != "modules"} for k, v in result.items()}, indent=2))
