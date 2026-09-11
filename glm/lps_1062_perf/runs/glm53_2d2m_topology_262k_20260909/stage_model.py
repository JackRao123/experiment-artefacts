"""Create a four-block FP8 HF checkpoint from explicitly mapped GLM-5.3 layers."""
import json, re, shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
ROOT = Path(__file__).resolve().parent
SRC = Path("/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6")
DST = ROOT/"model"
MAPPING = {0:0, 1:1, 3:2, 6:3}
def rename(name):
    m = re.match(r"model.layers.(\d+)\.(.*)", name)
    if m:
        old = int(m[1])
        if old == 1 and m[2].startswith(('self_attn.indexer.', 'self_attn.indexers_proj.')):
            return None
        return f"model.layers.{MAPPING[old]}.{m[2]}" if old in MAPPING else None
    return name if name.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")) else None
def main():
    DST.mkdir(exist_ok=True)
    assert not (DST/"_READY").exists(), "Refusing to overwrite completed model"
    config = json.loads((ROOT/"model-config.json").read_text())
    original = json.loads((SRC/"config.json").read_text())
    config["quantization_config"] = original["quantization_config"]
    config["quantization_config"]["modules_to_not_convert"] = [
        new for name in original["quantization_config"]["modules_to_not_convert"]
        if (new := rename(name)) is not None
    ]
    config["transformers_version"] = original["transformers_version"]
    (DST/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    index = json.loads((SRC/"model.safetensors.index.json").read_text())
    wanted = {}
    for name, shard in index["weight_map"].items():
        if rename(name) is not None:
            wanted.setdefault(shard, []).append(name)
    outmap, total = {}, 0
    for i, (shard, names) in enumerate(sorted(wanted.items())):
        outname = f"model-{i+1:05d}-of-{len(wanted):05d}.safetensors"
        with safe_open(SRC/shard, framework="pt", device="cpu") as f:
            tensors = {rename(name): f.get_tensor(name) for name in names}
        save_file(tensors, DST/outname, metadata={"format":"pt"})
        total += sum(t.numel()*t.element_size() for t in tensors.values())
        outmap.update({name:outname for name in tensors})
        print(outname, len(tensors), flush=True)
        del tensors
    (DST/"model.safetensors.index.json").write_text(json.dumps({"metadata":{"total_size":total},"weight_map":outmap},indent=2))
    for name in ("tokenizer.json","tokenizer_config.json","generation_config.json","chat_template.jinja"):
        if (SRC/name).exists(): shutil.copy2(SRC/name,DST/name)
    (DST/"provenance.json").write_text(json.dumps({"source":str(SRC),"layer_mapping":MAPPING,"tensor_count":len(outmap),"tensor_bytes":total},indent=2))
    (DST/"_READY").touch()
    print("READY",total,flush=True)
if __name__ == "__main__": main()
