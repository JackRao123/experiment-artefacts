#!/usr/bin/env bash
# usage: bash capture_and_analyze.sh <nsys-session> <tag>
# Captures one 262k step under an already-launched nsys session, exports sqlite, runs the analysis scripts.
set -x
SESSION=$1; TAG=$2
R=/root/.cache/user_artifacts/lps1062_bench/glm_nsys_gpu_metrics_262k_20260902
REPORT=glm53-$TAG-b300-262k-nvtx-all-ranks-gpu-metrics
PY=/root/.devbox-venvs/server/bin/python
nsys start --session=$SESSION --sample=process-tree --cpuctxsw=process-tree --backtrace=lbr --gpu-metrics-devices=all --gpu-metrics-frequency=10000 --gpuctxsw=true --output=$R/$REPORT --force-overwrite=true --stats=false
PYTHONPATH=$R $PY $R/capture_one.py --label $REPORT --seq-len 262144 --num-gpus 8 --output $R/$REPORT.json
nsys stop --session=$SESSION
echo CAPTURE_DONE
nsys export --type sqlite --force-overwrite true --output $R/$TAG.sqlite $R/$REPORT.nsys-rep
echo EXPORT_DONE
cd $R
$PY nsys_attrib.py $TAG.sqlite --out analysis_$TAG --window nvtx:forward_backward --gpu-metrics > analysis_$TAG.log 2>&1
$PY sync_callchains.py $TAG.sqlite --out analysis_$TAG >> analysis_$TAG.log 2>&1
$PY sync_arrivals.py $TAG.sqlite --out analysis_$TAG >> analysis_$TAG.log 2>&1
$PY category_by_range.py $TAG.sqlite --out analysis_$TAG >> analysis_$TAG.log 2>&1
$PY range_imbalance.py $TAG.sqlite --out analysis_$TAG >> analysis_$TAG.log 2>&1
echo ANALYSIS_DONE
