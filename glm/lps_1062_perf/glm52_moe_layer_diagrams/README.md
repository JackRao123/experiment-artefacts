# GLM-5.2 single MoE decoder layer

This directory contains editable D2 architecture diagrams for one representative GLM-5.2 sparse decoder layer.

The diagram uses a full-indexer layer, such as 1-based layer 7. GLM-5.2's shared-indexer layers reuse DSA top-k token indices from the preceding full-indexer layer; that alternate path is also labeled.

## Split left-to-right diagrams

The `split/` directory contains three standalone diagrams:

- `glm-5.2-layer-overview`: complete decoder-layer input-to-output flow
- `glm-5.2-self-attention`: expanded `self_attn` architecture
- `glm-5.2-moe`: expanded `mlp` / MoE architecture

Each is available as editable `.d2` source and rendered `.svg`, `.png`, and `.pdf` files.

## Render split diagrams

```bash
cd split
d2 --layout elk --theme 0 --pad 40 glm-5.2-layer-overview.d2 glm-5.2-layer-overview.pdf
d2 --layout elk --theme 0 --pad 40 glm-5.2-self-attention.d2 glm-5.2-self-attention.pdf
d2 --layout elk --theme 0 --pad 40 glm-5.2-moe.d2 glm-5.2-moe.pdf
```

The original combined diagram remains in the parent directory.

The diagram assumes batch size `B = 1` and omits the batch axis. The input and output tensors both have shape `[S, 6144]`. Residual connections preserve the shape, but the output values differ from the input values.

Learned components are labeled with their exact Hugging Face `nn.Module` attribute paths, relative to `model.layers.<layer_idx>`. Tensor-only operations retain descriptive names because they are functions or inline operations rather than separate modules.

Architecture values come from the following sources, accessed on 2026-08-21:

- [zai-org/GLM-5.2 config.json](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [Hugging Face Transformers GlmMoeDsa implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)
