"""Validate accepted artifact integrity and protocol; no test-suite files changed."""
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
validation={}
for topology in ['cp8ep8','cp8ep1','cp1ep1']:
    folder=ROOT/topology/'result'
    b=json.loads((folder/'benchmark.json').read_text())
    s=json.loads((folder/'summary.json').read_text())
    assert Counter(w['phase'] for w in b['windows'])==dict(warmup=3,control=5,memory_profile=1,runtime_profile=1)
    assert b['seq_len']==131072
    assert b['final_status']['step']-b['initial_status']['step']==10
    assert len(list((folder/'runtime').glob('*.json')))==b['num_gpus']
    for relative,expected in s['artifacts'].items():
        p=folder/relative
        assert p.stat().st_size==expected['bytes']
        with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected['sha256']
    memories=[]
    for p in (folder/'memory').glob('*.pickle'):
        with p.open('rb') as f:snapshot=pickle.load(f)
        events=sum(len(t) for t in snapshot['device_traces'])
        assert events>0 and len(snapshot['segments'])>0
        memories.append(dict(file=p.name,segments=len(snapshot['segments']),history_events=events))
    analysis=json.loads((folder/'analysis/summary.json').read_text())
    assert len(analysis)==b['num_gpus']
    for rank,data in analysis.items():assert len(data['kernels'])>0
    validation[topology]=dict(runtime_ranks=len(analysis),memory=memories,integrity='passed')
    inventories=list((folder/'timings').glob('weights.rank*.json'))
    assert len(inventories)==b['num_gpus']
    expected=128 if topology=='cp8ep8' else 1024
    for p in inventories:
        weights=json.loads(p.read_text())
        assert len(weights)==expected
        assert all(w['dtype']=='torch.bfloat16' and not w['quantized_payload'] for w in weights)
    validation[topology]['resident_bf16_weights_per_rank']=expected
    if topology=='cp1ep1':
        kernels=analysis['0']['kernels']
        fused=sum(int(k['calls']) for k in kernels if 'indexer_fwd' in k['name'])
        fallback=sum(int(k['calls']) for k in kernels if 'cutlass_80_simt_sgemm_128x64_8x5_nn_align1' in k['name'])
        nccl=sum(int(k['calls']) for k in kernels if 'nccl' in k['name'].lower())
        assert fused>0 and fallback==0 and nccl==0
        validation[topology]['kernel_verification']=dict(cudnn_indexer_calls=fused,old_fallback_calls=fallback,nccl_calls=nccl)
(ROOT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
