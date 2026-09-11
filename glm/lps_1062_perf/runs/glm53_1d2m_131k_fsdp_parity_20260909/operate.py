"""Explicit lifecycle phases for matched numerical experiments."""
import argparse
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/root/glm53-131k-fsdp-parity-20260909'
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('ddp','fsdp','fsdp_grouped'));ap.add_argument('phase',choices=('start','wait','run','stop'));a=ap.parse_args()
env=f'export PARITY_FSDP={int(a.mode.startswith("fsdp"))} PARITY_GROUPED_MM={int(a.mode=="fsdp_grouped")} BT_TRAINER_CONFIG_PATH={REMOTE}/{a.mode}/trainer-config.json BT_TRAINER_SERVER_CONFIG_PATH=/root/glm53-main-262k-20260909/trainer-server-config.json LAYER_TIMING_DIR={REMOTE}/{a.mode}/timings BT_FREEZE_GC_AFTER_WARMUP=1 NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; '
if a.phase=='run':
    cmd=f'set -eo pipefail; /root/.devbox-venvs/server/bin/python {REMOTE}/profile_driver.py --label parity-{a.mode} --seq-len 131072 --datums 1 --num-gpus 8 --warmup-repeats 3 --control-repeats 5 2>&1 | tee {REMOTE}/{a.mode}/driver.log; cp {REMOTE}/.devbox_up/trainer_srun.log {REMOTE}/{a.mode}/trainer.log'
else:
    script={'start':'start_trainer','wait':'wait_trainer_health','stop':'stop_trainer'}[a.phase]
    cmd=env+f'bash {REMOTE}/.devbox_up/{script}.sh'
subprocess.run(['ssh','tj-32vj99q',cmd],check=True)
if a.phase=='run':
    subprocess.run(['scp','-C',f'tj-32vj99q:{REMOTE}/results/parity-{a.mode}.json',str(ROOT/a.mode/'benchmark.json')],check=True)
    subprocess.run(['scp','-r','-C',f'tj-32vj99q:{REMOTE}/{a.mode}/timings',str(ROOT/a.mode/'timings')],check=True)
