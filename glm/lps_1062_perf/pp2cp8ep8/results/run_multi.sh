#!/bin/bash
# 8-GPU PCIe/host-DRAM contention driver. Pins each worker to 8 cores on its
# GPU's local NUMA node. Usage: run_multi.sh <seconds-per-mode>
set -u
SECS=${1:-20}
BIN=/root/bw_multi
NGPU=$(nvidia-smi -L | wc -l)

# GPU -> NUMA node map (bus id like 00000000:1A:00.0 -> 0000:1a:00.0)
declare -a NODE
for i in $(seq 0 $((NGPU-1))); do
  bus=$(nvidia-smi -i $i --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-F' 'a-f' | sed 's/^0000//')
  NODE[$i]=$(cat /sys/bus/pci/devices/${bus%.*}.0/numa_node 2>/dev/null || echo -1)
done
echo "GPU->NUMA: ${NODE[@]}"

# core ranges: node0 = 0-63, node1 = 64-127 (physical cores; SMT twins 128+)
core_range() { # gpu index -> 8 dedicated cores on its local node
  local i=$1 n=${NODE[$1]}
  local base=$(( n == 1 ? 64 : 0 ))
  echo "$((base + i*8))-$((base + i*8 + 7))"
}

run_mode() { # <mode> <gpuset: "0" or "all">
  local mode=$1 set=$2 pids=()
  local out="/tmp/bw_${mode}_${set}.$$"
  rm -f $out
  if [ "$set" = "0" ]; then
    CUDA_VISIBLE_DEVICES=0 taskset -c $(core_range 0) $BIN $mode $SECS gpu0 >> $out &
    pids+=($!)
  else
    for i in $(seq 0 $((NGPU-1))); do
      CUDA_VISIBLE_DEVICES=$i taskset -c $(core_range $i) $BIN $mode $SECS gpu$i >> $out &
      pids+=($!)
    done
  fi
  wait "${pids[@]}"
  echo "== $mode ($set GPU) =="
  sort $out
  awk '{s+=$3} END {printf "AGGREGATE %s: %.1f GB/s (n=%d)\n","'$mode'",s,NR}' $out
  rm -f $out
}

echo "--- single-GPU baseline ---"
run_mode d2h 0
run_mode h2d 0
run_mode bidi 0
echo "--- all-$NGPU concurrent ---"
run_mode d2h all
run_mode h2d all
run_mode bidi all
echo DONE
