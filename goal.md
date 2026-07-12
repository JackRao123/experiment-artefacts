These are supposed to survive context window compaction.


INFORMATION
- Read the devbox skill.
- SSH into the 3x8xB200 devbox, thats where we were developing before. but also we now have a 4x8xB200 devbox, but this is on a different cluster so the two boxes don't share disk.
- Use branch jack-nemo3super-131k.
- Dataset: nvidia/ChatQA2-Long-SFT-data → NarrativeQA_131072 config, and pirate-ultrachat-10k should already be downloaded. Although for context OOM tests you can just use synthetic data? 
- You can spawn subagents for context-draining tasks if you want to.


GOAL
- Determine a suitable golden_config for nemotron 3 super, 131k context, and tweak until it works and doesn't OOM. LoRA SFT.
- Once you are done create a PR that requests to merge into jack-nemo3super.
- Ensure the changes you need to make are retained inside the /trainers repo, because I want this to work in prod, not just random ad hoc stuff that will work here but is not persisted. Also retain the scripts you use for SFT for reproducibility inside the /trainers repo and include them in the PR.

COMMON LLM PITFALLS
1. Not realising you are on a multi-node box. 
2. GPU memory frozen on a multi-node box, it might be the case that another node has crashed and now the node you are on is hanging, waiting for the other node, full of memory appearing to be doing work but actually just hanging.
3. Not having git credentials on a box. On my laptop i have `gh auth token`. Feel free to take it onto the box.
4. Trying to manually install the environment defined in /trainers, instead of just running uv sync or shell scripts or similar.
5. Seeing that there's already a sampler/trainer running on a node and being scared to kill it. 99% of the time you can kill it without asking because its just a leftover process. Otherwise I would have told you to be careful.

 