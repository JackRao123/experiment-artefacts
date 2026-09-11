"""Run one explicit lifecycle phase; waiter checkpoints return to the operator."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/glm53-131k-fsdp-cp8ep1-20260909'
HOST = 'tj-32vj99q'
ap = argparse.ArgumentParser()
ap.add_argument('topology', choices=('cp8ep1', 'cp8ep8', 'cp1ep1'))
ap.add_argument('phase', choices=('start', 'wait', 'benchmark', 'stability', 'stop'))
a = ap.parse_args()
gpus = 1 if a.topology == 'cp1ep1' else 8
d = f'{REMOTE}/.devbox_up'
label = f'{a.topology}-identity-sort'
env = f'export BT_TRAINER_CONFIG_PATH={REMOTE}/{a.topology}/trainer-config.json BT_TRAINER_SERVER_CONFIG_PATH=/root/glm53-main-262k-20260909/trainer-server-config.json LAYER_TIMING_DIR={REMOTE}/{a.topology}/result_timings BT_FREEZE_GC_AFTER_WARMUP=1 NUM_GPUS={gpus} CUDA_VISIBLE_DEVICES={",".join(map(str,range(gpus)))};'
if a.phase in ('start', 'wait', 'stop'):
    cmd = env + f' bash {d}/{dict(start="start_trainer",wait="wait_trainer_health",stop="stop_trainer")[a.phase]}.sh'
else:
    stability = a.phase == 'stability'
    suffix = 'stability' if stability else 'result'
    actual_label = label + ('-stability' if stability else '')
    args = '--warmup-repeats 0 --control-repeats 20' if stability else '--warmup-repeats 3 --control-repeats 5 --memory-profile --runtime-profile'
    cmd = f'set -eo pipefail; /root/.devbox-venvs/server/bin/python {REMOTE}/profile_driver.py --label {actual_label} --seq-len 131072 --datums 1 --num-gpus {gpus} {args} 2>&1 | tee {REMOTE}/{a.topology}/{suffix}.log; cp {d}/trainer_srun.log {REMOTE}/{a.topology}/{suffix}.trainer.log'
subprocess.run(['ssh',HOST,cmd],check=True)
if a.phase == 'benchmark':
    subprocess.run(['python3',str(ROOT/'collect.py'),a.topology,'result',label],check=True)
if a.phase == 'stability':
    subprocess.run(['scp','-C',f'{HOST}:{REMOTE}/results/{label}-stability.json',str(ROOT/a.topology/'stability.json')],check=True)
    subprocess.run(['scp','-r','-C',f'{HOST}:{REMOTE}/{a.topology}/result_timings',str(ROOT/a.topology/'stability_timings')],check=True)
