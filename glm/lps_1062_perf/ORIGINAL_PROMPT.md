Overnight task, work autonomously

Use this as the baseline. You will start from here. branch: revert-1064-vram-release
Create two branches, branching off this branch.
1. lps1062-deepep-test
2. lps1062-hybridep-test

Let 'flex' = {deepep|hybridep}.

You will, on each of those two branches, enable the Flex.
1. You should get it working first. You should use a debug model. A debug model means you take GLM 5.2 and you shrink it down to one dense layer and one MoE layer. Don't touch any of the other dimensions or characteristics, because that might mess it up, for example hidden dimension. Just turn down the number of layers, okay? This way, your iteration is fast because you don't have to wait the entire 20-minute startup time of the full model. Instead, it can only take one minute or something. Because you have to test EP, you might want to use configs of two, four, or eight GPUs, for example. 
Iterate like this to do optimisations, fixes, etc. Ensure that the flex is working to its full potential. You should also record traces and have subagents analyze these traces to find inefficiencies/bottlenecks that are going wrong related to the flex. The scope is just to flex - don't worry about other unrelated optimisations.

2. When you think you have it working, run it on the full GLM 5.2 model, and then use the benchmark driver (profile_driver_new.py). If you run into issues, go back to using the debug model in (1) and use the full model only for e2e validation.

3. Goal is, by morning, have deepEP and hybridEP working and also record a trace for me, and also compare their TPS to the baseline. Give me a report.



Rules: 
1. All work you do must get persisted to on one of those two branches, or in /Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf folder if its some artefact. When you make code changes to the trainer, you must commit and push it always, so I can see your progress as you go.
2. Keep a notebook-like log in that folder and update it as you go.
3. Here is a devbox, ssh tj-q4grmdq. I was doing some stuff on it before, but dont worry about it, its ok, its yours now. Follow devbox-up guide and rules.
4. Periodically re-read this so you don't forget the goal.
