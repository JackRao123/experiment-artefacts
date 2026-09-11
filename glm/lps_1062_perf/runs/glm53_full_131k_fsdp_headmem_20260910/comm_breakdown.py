"""Use Kineto process-group metadata rather than guessing collective ownership."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args()
    trace=next((a.folder/'runtime').glob('*rank0*.json'))
    events=json.loads(trace.read_text())['traceEvents']
    comm={}
    for e in events:
        if e.get('name')=='record_param_comms':
            args=e.get('args',{})
            comm[args.get('External id')]=args
    groups=defaultdict(lambda:dict(calls=0,input_bytes=0,output_bytes=0,kernel_calls=0,kernel_ms=0.))
    def key(args):return f"{args.get('Process Group Description')} / {args.get('Collective name')}"
    for args in comm.values():
        if args.get('Group size',1)<=1:continue
        value=groups[key(args)];value['calls']+=1
        size={'BFloat16':2,'Half':2,'Float':4,'Double':8,'Long':8,'Int':4,'Byte':1}.get(args.get('dtype'),0)
        value['input_bytes']+=args.get('In msg nelems',0)*size
        value['output_bytes']+=args.get('Out msg nelems',0)*size
    for e in events:
        if e.get('cat')!='kernel' or 'nccl' not in e.get('name','').lower():continue
        args=comm.get(e.get('args',{}).get('External id'))
        if not args:continue
        value=groups[key(args)];value['kernel_calls']+=1;value['kernel_ms']+=e.get('dur',0)/1000
    out=a.folder/'analysis';out.mkdir(exist_ok=True)
    (out/'communication.json').write_text(json.dumps(dict(groups),indent=2)+'\n')
    print(json.dumps(dict(groups),indent=2))
if __name__=='__main__':main()
