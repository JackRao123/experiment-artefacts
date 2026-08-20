# Goal — PP2/CP8/EP8 @131k bring-up (2026-08-10 night)

Verbatim prompt from Jack (2026-08-10 ~20:35 PT), saved per instruction:

---

See: /Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf

Inside here are a lot of artifacts from when we were experimenting and trying to optimize GLM 5.2 trainer throughput and MFU. However, a lot of these did not end up working and are just slop.

What I want to do is start from scratch (tip of trainers) and start with a different direction:

My overall goal is to maximise MFU and TPU/GPS for GLM 5.2 training.
The current task: your current goal is to get this config working - PP2,CP8,EP8, 131k seqlen. If theres other code changes you need to make or implement etc, you can make them.

Here are some notes and thoughts i was brainstorming. 1,2,3 are instructions, 4 is more just a guide and my logical reasoning.

"""
Constraints

1. In `glm52_dsa.py` there is no entry for PP2. However this is easy to solve because all this is is telling Megatron how the layers are split between the PP ranks. I think we can just get AI agent to figure out the split, somewhat equally across both. There is something about where the topk selection is shared across a group, so we have to make PP rank 1 start on a layer at the start of the group, but this should be easy to figure out just by looking at the config.
2. Same size microbatch required between PP ranks. This is required per call - not globally but we can probably just fix it globally to 131k because we use CP/THD packing. I think we can just pack every partition to 131K. We can be fine with the lower-digit percentage waste because most of the time the partitions will be packed fully to nearly 131K. I think it's fine to just pad to 131k.
3. Test if PP>1 CP>1 lora adapter export, works. Do this with a minimal model, don't need to spawn up the entire GLM.
4. How many microbatches can we fit? I think intuitively we can fit 262k very comfortably on EP16 and CP16. If we just decrease the sequence length, we halve it, and then we also divide the sequence length by two, then that means we still have the same amount of tokens per GPU. As for the EP part we have double the amount of exports per layer per GPU but then we have half the amount of layers per GPU. That means it cancels out. If we are able to support a micro-batch of size 262k on EP16 and CP16, we should be able to support two micro-batches of 131k on EP8, CP8, PP2.
    - Because we have full activations recomputation, the only thing we have to be wary of is the spike in intermediate activations when going through each layer. I think because we originally were storing 78 layers' worth of checkpointed activations, like the checkpointed ones, (in the above calculation I'm talking about the number of checkpointed activations) - surely the intermediate activations of a single layer, which are all the excess we will materialize at any given time and then discard after that layer, surely that will not make us OOM.
"""

Use /devbox-up skill to spawn nodes.
Make sure when you're testing launches that we haven't done before (eg the PP+CP thing) that you set short, bounded timeouts (5-10m), and you check on it periodically so that you don't end up waiting while something is clearly deadlocked.

You are pauli, you are the manager. You are a claude fable 1m.
Gibbs, volta, and dedekind are your kimi k3 1m subordinates, they all have fresh sessions. They will do all the work and experimentation for you. If you don't need that many subordinates for this task, then you don't have to use them all.
Also for anything that is a simple disposable task that you don't want to bloat memory with, you (or they) can just spawn a NEW subagent.

Work autonomously. Also once you get it working, use /Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/tools/profile_driver_new.py to record a new profile of it, and measure TPS and MFU.

Also in notebook.md, write down a log of the things you do over the night, and save artefacts into a folder in that experiment artefacts/glm/lps_1062_perf folder.

Save this prompt somewhere in your folder as your goal so you can look at it in the future.
