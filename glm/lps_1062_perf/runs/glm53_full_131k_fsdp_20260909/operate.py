"""Full-model lifecycle and protocol. No automated health polling loop."""
import argparse
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/root/glm53-full-131k-fsdp-20260909'
ap=argparse.ArgumentParser();ap.add_argument('cp',type=int,choices=(8,4,2));ap.add_argument('phase',choices=('start','wait','benchmark','stop'));a=ap.parse_args()
topology=f'cp{a.cp}ep1';d=f'{REMOTE}/.devbox_up';label=f'glm53-full-fsdp-{topology}'
env=f'export BT_TRAINER_CONFIG_PATH={REMOTE}/{topology}/trainer-config.json BT_TRAINER_SERVER_CONFIG_PATH=/root/glm53-main-262k-20260909/trainer-server-config.json LAYER_TIMING_DIR={REMOTE}/{topology}/result_timings BT_FREEZE_GC_AFTER_WARMUP=1 NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; '
if a.phase=='benchmark':
    cmd=f'set -eo pipefail; /root/.devbox-venvs/server/bin/python {REMOTE}/profile_driver.py --label {label} --seq-len 131072 --datums {8//a.cp} --num-gpus 8 --warmup-repeats 3 --control-repeats 5 --memory-profile --runtime-profile 2>&1 | tee {REMOTE}/{topology}/result.log; cp {d}/trainer_srun.log {REMOTE}/{topology}/result.trainer.log'
else:
    script={'start':'start_trainer','wait':'wait_trainer_health','stop':'stop_trainer'}[a.phase]
    cmd=env+f'bash {d}/{script}.sh'
result=subprocess.run(['ssh','tj-32vj99q',cmd],check=a.phase!='wait')
if result.returncode:raise SystemExit(result.returncode)
if a.phase=='benchmark':subprocess.run(['python3',str(ROOT/'collect.py'),str(a.cp)],check=True)
