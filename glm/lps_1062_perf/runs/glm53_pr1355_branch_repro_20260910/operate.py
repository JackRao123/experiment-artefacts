"""Run clean branch code through generated devbox lifecycle scripts."""

import argparse
import shlex
import subprocess


REMOTE = '/root/glm53-pr1355-repro-20260910'
VARIANTS = ('debug-cp8', 'debug-cp4', 'debug-grouped-cp8', 'full-cp8', 'full-cp4')
parser = argparse.ArgumentParser()
parser.add_argument('variant', choices=VARIANTS)
parser.add_argument('phase', choices=('start', 'wait', 'benchmark', 'stop'))
parser.add_argument('--controls', type=int, default=5)
parser.add_argument('--seq-len', type=int, default=131072)
parser.add_argument('--profiles', action='store_true')
args = parser.parse_args()
cp = 4 if args.variant.endswith('cp4') else 8
env = {
    'BT_TRAINER_CONFIG_PATH': f'{REMOTE}/{args.variant}/trainer-config.json',
    'BT_TRAINER_SERVER_CONFIG_PATH': '/root/glm53-main-262k-20260909/trainer-server-config.json',
    'BT_PROFILE_OUTPUT_DIR': f'{REMOTE}/{args.variant}/profiles',
    'BT_EXPERIMENTAL_FSDP': '1',
    'BT_FSDP_PREFETCH': '0' if cp == 4 else '1',
    'BT_FSDP_PERSISTENT_BUFFERS': '1',
    'BT_FSDP_FAST_LOAD': '1',
    'BT_FSDP_GROUPED_MM': '1' if 'grouped' in args.variant else '0',
    'BT_MEMORY_EFFICIENT_LM_HEAD': '1' if cp == 4 else '0',
    'BT_LM_HEAD_SEQ_CHUNK': '2048' if cp == 4 else '4096',
    'BT_FREEZE_GC_AFTER_WARMUP': '1',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,garbage_collection_threshold:0.95',
    'NUM_GPUS': '8',
    'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
}
prefix = 'export ' + ' '.join(f'{key}={shlex.quote(value)}' for key, value in env.items()) + '; '
if args.phase == 'benchmark':
    command = (
        f'set -eo pipefail; /root/.devbox-venvs/server/bin/python {REMOTE}/profile_driver.py '
        f'--label pr1355-{args.variant} --seq-len {args.seq_len} --datums {8 // cp} '
        f'--num-gpus 8 --warmup-repeats 3 --control-repeats {args.controls} '
        + ('--memory-profile --runtime-profile ' if args.profiles else '')
        + f'2>&1 | tee {REMOTE}/{args.variant}/driver.log'
    )
else:
    script = {'start': 'start_trainer', 'wait': 'wait_trainer_health', 'stop': 'stop_trainer'}[args.phase]
    command = prefix + f'bash {REMOTE}/.devbox_up/{script}.sh'
result = subprocess.run(['ssh', '-S', 'none', 'tj-32vj99q', command])
raise SystemExit(result.returncode)
