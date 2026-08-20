// Multi-GPU PCIe contention benchmark worker: hammers one direction for a
// fixed duration on device 0 (select GPU via CUDA_VISIBLE_DEVICES), prints
// one line: "<tag> <mode> <GB/s>". Launch 8 concurrently to measure
// aggregate host-DRAM/PCIe contention.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static const size_t BUF = 4ULL<<30;
static const size_t CHUNK = 1ULL<<30;

double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 4) { printf("usage: bw_multi <d2h|h2d|bidi> <seconds> <tag>\n"); return 1; }
  std::string mode = argv[1];
  double secs = atof(argv[2]);
  const char* tag = argv[3];

  CHECK(cudaSetDevice(0));
  char *d_a,*d_b,*h_a,*h_b;
  CHECK(cudaMalloc(&d_a,BUF)); CHECK(cudaMallocHost(&h_a,BUF));
  CHECK(cudaMemset(d_a,1,BUF)); memset(h_a,2,BUF);
  if (mode=="bidi") { CHECK(cudaMalloc(&d_b,BUF)); CHECK(cudaMallocHost(&h_b,BUF));
                      CHECK(cudaMemset(d_b,3,BUF)); memset(h_b,4,BUF); }
  cudaStream_t s1,s2; CHECK(cudaStreamCreate(&s1)); CHECK(cudaStreamCreate(&s2));

  // warmup
  CHECK(cudaMemcpyAsync(h_a,d_a,CHUNK,cudaMemcpyDeviceToHost,s1));
  CHECK(cudaStreamSynchronize(s1));

  double t0 = now(); size_t bytes = 0;
  while (now()-t0 < secs) {
    for (size_t off=0; off<BUF; off+=CHUNK) {
      if (mode=="d2h") {
        CHECK(cudaMemcpyAsync(h_a+off,d_a+off,CHUNK,cudaMemcpyDeviceToHost,s1)); bytes+=CHUNK;
      } else if (mode=="h2d") {
        CHECK(cudaMemcpyAsync(d_a+off,h_a+off,CHUNK,cudaMemcpyHostToDevice,s1)); bytes+=CHUNK;
      } else { // bidi
        CHECK(cudaMemcpyAsync(h_a+off,d_a+off,CHUNK,cudaMemcpyDeviceToHost,s1));
        CHECK(cudaMemcpyAsync(d_b+off,h_b+off,CHUNK,cudaMemcpyHostToDevice,s2));
        bytes+=2*CHUNK;
      }
    }
    CHECK(cudaStreamSynchronize(s1));
    if (mode=="bidi") CHECK(cudaStreamSynchronize(s2));
  }
  double dt = now()-t0;
  printf("%s %s %.1f\n", tag, mode.c_str(), bytes/dt/1e9);
  return 0;
}
