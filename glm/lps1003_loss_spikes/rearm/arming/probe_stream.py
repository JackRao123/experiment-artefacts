import torch, json
import cuda.bindings.driver as cuda
import torch.profiler as P

a = torch.zeros(1<<20, device="cuda")
torch.cuda.synchronize()

cs = torch.cuda.current_stream()
print("outside raw:", hex(cs.cuda_stream))
h = cuda.CUstream(cs.cuda_stream)
print("int(CUstream):", int(h), hex(int(h)))
h2 = cuda.CUstream(cs.cuda_stream)
print("int(CUstream) new obj:", int(h2), hex(int(h2)))
es = torch.cuda.ExternalStream(int(h))
print("es.cuda_stream:", hex(es.cuda_stream))

with P.profile(activities=[P.ProfilerActivity.CPU, P.ProfilerActivity.CUDA]) as prof:
    a.fill_(1.0)
    with torch.cuda.stream(es):
        print("inside es ctx raw:", hex(torch.cuda.current_stream().cuda_stream))
        a.fill_(2.0)
    es2 = torch.cuda.ExternalStream(int(cuda.CUstream(torch.cuda.current_stream().cuda_stream)))
    with torch.cuda.stream(es2):
        a.fill_(3.0)
    s = torch.cuda.Stream()
    print("real created stream raw:", hex(s.cuda_stream))
    with torch.cuda.stream(s):
        a.fill_(4.0)
    torch.cuda.synchronize()
prof.export_chrome_trace("/root/arming/results/probe_stream.json")
evs = json.load(open("/root/arming/results/probe_stream.json"))["traceEvents"]
for e in sorted([e for e in evs if e.get("cat")=="kernel"], key=lambda x: x["ts"]):
    print("kernel stream", e["args"].get("stream"), e["name"][:60])
