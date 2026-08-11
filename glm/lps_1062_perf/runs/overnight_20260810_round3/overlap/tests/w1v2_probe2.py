import os, sys, tempfile, types
sys.path.insert(0, os.path.expanduser('~/Documents/trainers/server/vendor/megatron-bridge/3rdparty/Megatron-LM'))
_stub = types.ModuleType('megatron.core.transformer.moe.ops.paged_stash')
_stub.GLOBAL_BLOCK_SIZE = 128; _stub.paged_stash_copy_kernel = None; _stub.paged_stash_pop_kernel = None
sys.modules.setdefault('megatron.core.transformer.moe.ops.paged_stash', _stub)
import torch, torch.distributed as dist, torch.multiprocessing as mp
import megatron.core.tensor_parallel.mappings as mappings

ORDER = []
_orig_bwd = mappings._AllToAllDeferredWait.backward
@staticmethod
def _logged_bwd(ctx, *grad_output):
    ORDER.append(f'a2a.bwd seq={grad_output[0].grad_fn._sequence_nr() if grad_output[0].grad_fn else "?"}')
    return _orig_bwd(ctx, *grad_output)
# can't staticmethod-patch easily; instead tag via ctx: patch forward to record group identity
class Probe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, p): return t * 1.5, p * 2.5
    @staticmethod
    def backward(ctx, gt, gp):
        ORDER.append('probe.bwd')
        return gt, gp

def worker(rank, ws, store):
    dist.init_process_group('gloo', store=dist.FileStore(store, ws), rank=rank, world_size=ws)
    g = dist.group.WORLD
    g2 = dist.new_group(dist.get_process_group_ranks(g))
    IN = {0: [3, 5], 1: [2, 4]}; OUT = {0: [3, 2], 1: [5, 4]}
    gen = torch.Generator().manual_seed(rank)
    t = torch.randn(sum(IN[rank]), 4, generator=gen, requires_grad=True)
    p = torch.randn(sum(IN[rank]), generator=gen, requires_grad=True)
    ot = mappings.all_to_all_deferred(g, t, OUT[rank], IN[rank])
    op = mappings.all_to_all_deferred(g2, p, OUT[rank], IN[rank])
    mappings.wait_deferred_a2a(ot); mappings.wait_deferred_a2a(op)
    seq_t = ot.grad_fn._sequence_nr(); seq_p = op.grad_fn._sequence_nr()
    # instrument: wrap each a2a node's backward via a pre-hook is not available;
    # instead wrap outputs with identity Functions that log in bwd order
    class Tag(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, name):
            ctx.name = name
            return x
        @staticmethod
        def backward(ctx, g):
            ORDER.append(f'{ctx.name}.bwd')
            return g, None
    ot2 = Tag.apply(ot, 'tokens-a2a')
    op2 = Tag.apply(op, 'probs-a2a')
    ht, hp = Probe.apply(ot2, op2)
    (ht.sum() + hp.sum()).backward()
    if rank == 0:
        print(f'seqs: tokens-a2a={seq_t} probs-a2a={seq_p}', flush=True)
        print(f'bwd order: {ORDER}', flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(2, os.path.join(tmp, 's')), nprocs=2, join=True)
