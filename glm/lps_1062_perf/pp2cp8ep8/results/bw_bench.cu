// PCIe D2H/H2D bandwidth microbenchmark for the LPS-1062 activation-offload
// feasibility question. Pinned + pageable, chunk sweep, bidirectional, and
// copy-under-compute. No deps beyond CUDA runtime.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static const size_t BUF = 4ULL<<30; // 4 GiB per buffer

__global__ void spin_fma(float* out, long long iters) {
  float a = (threadIdx.x + blockIdx.x) * 1.000001f;
  float b = 1.0000001f, c = 0.0000001f;
  for (long long i = 0; i < iters; i++) {
    a = fmaf(a, b, c); a = fmaf(a, b, c); a = fmaf(a, b, c); a = fmaf(a, b, c);
  }
  if (a == 12345.678f) out[threadIdx.x] = a; // never true; defeats DCE
}

double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Copy `bytes` in `chunk`-sized pieces, `reps` times, on stream s. Returns GB/s.
double bench_copy(void* dst, const void* src, size_t bytes, size_t chunk,
                  cudaMemcpyKind kind, cudaStream_t s, int reps) {
  for (size_t off=0; off<bytes; off+=chunk) // warmup pass
    CHECK(cudaMemcpyAsync((char*)dst+off,(const char*)src+off,std::min(chunk,bytes-off),kind,s));
  CHECK(cudaStreamSynchronize(s));
  double t0 = now();
  for (int r=0;r<reps;r++)
    for (size_t off=0; off<bytes; off+=chunk)
      CHECK(cudaMemcpyAsync((char*)dst+off,(const char*)src+off,std::min(chunk,bytes-off),kind,s));
  CHECK(cudaStreamSynchronize(s));
  return (double)bytes*reps/(now()-t0)/1e9;
}

int main() {
  int dev=0; CHECK(cudaSetDevice(dev));
  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p,dev));
  printf("GPU: %s  SMs=%d  cc=%d.%d\n", p.name, p.multiProcessorCount, p.major, p.minor);

  char *d_a,*d_b,*h_pin_a,*h_pin_b;
  CHECK(cudaMalloc(&d_a,BUF)); CHECK(cudaMalloc(&d_b,BUF));
  CHECK(cudaMallocHost(&h_pin_a,BUF)); CHECK(cudaMallocHost(&h_pin_b,BUF));
  CHECK(cudaMemset(d_a,1,BUF)); CHECK(cudaMemset(d_b,2,BUF));
  memset(h_pin_a,3,BUF); memset(h_pin_b,4,BUF);
  cudaStream_t s1,s2,sk;
  CHECK(cudaStreamCreate(&s1)); CHECK(cudaStreamCreate(&s2));
  CHECK(cudaStreamCreateWithPriority(&sk,cudaStreamNonBlocking,0));
  float* d_out; CHECK(cudaMalloc(&d_out,1024*sizeof(float)));

  const int REPS = 8; // 32 GiB moved per measurement
  size_t chunks[] = {64ULL<<20, 256ULL<<20, 1ULL<<30, BUF};
  const char* cn[]  = {"64MB","256MB","1GB","4GB"};

  printf("\n-- pinned, unidirectional (GB/s) --\n");
  printf("%-8s %8s %8s\n","chunk","D2H","H2D");
  for (int i=0;i<4;i++) {
    double d2h = bench_copy(h_pin_a,d_a,BUF,chunks[i],cudaMemcpyDeviceToHost,s1,REPS);
    double h2d = bench_copy(d_b,h_pin_b,BUF,chunks[i],cudaMemcpyHostToDevice,s1,REPS);
    printf("%-8s %8.1f %8.1f\n",cn[i],d2h,h2d);
  }

  printf("\n-- pinned, bidirectional simultaneous (1GB chunks) --\n");
  {
    size_t chunk = 1ULL<<30;
    // warmup both
    bench_copy(h_pin_a,d_a,BUF,chunk,cudaMemcpyDeviceToHost,s1,1);
    bench_copy(d_b,h_pin_b,BUF,chunk,cudaMemcpyHostToDevice,s2,1);
    double t0 = now();
    for (int r=0;r<REPS;r++)
      for (size_t off=0; off<BUF; off+=chunk) {
        CHECK(cudaMemcpyAsync(h_pin_a+off,d_a+off,chunk,cudaMemcpyDeviceToHost,s1));
        CHECK(cudaMemcpyAsync(d_b+off,h_pin_b+off,chunk,cudaMemcpyHostToDevice,s2));
      }
    CHECK(cudaStreamSynchronize(s1)); CHECK(cudaStreamSynchronize(s2));
    double dt = now()-t0;
    printf("aggregate %.1f GB/s (D2H+H2D together; per-direction ~%.1f if symmetric)\n",
           2.0*BUF*REPS/dt/1e9, BUF*REPS/dt/1e9);
  }

  printf("\n-- copies while SMs are saturated with FMA kernel --\n");
  {
    // calibrate kernel to ~6s
    long long iters = 200000000LL;
    spin_fma<<<p.multiProcessorCount*4,256,0,sk>>>(d_out,iters/50);
    CHECK(cudaStreamSynchronize(sk));
    double tk0=now();
    spin_fma<<<p.multiProcessorCount*4,256,0,sk>>>(d_out,iters/10);
    CHECK(cudaStreamSynchronize(sk));
    double unit = (now()-tk0)*10.0/iters; // sec per iter
    long long target = (long long)(6.0/unit);
    // kernel alone, timed
    double ta0=now();
    spin_fma<<<p.multiProcessorCount*4,256,0,sk>>>(d_out,target);
    CHECK(cudaStreamSynchronize(sk));
    double alone = now()-ta0;
    // kernel + concurrent D2H
    double tb0=now();
    spin_fma<<<p.multiProcessorCount*4,256,0,sk>>>(d_out,target);
    double d2h_busy = bench_copy(h_pin_a,d_a,BUF,1ULL<<30,cudaMemcpyDeviceToHost,s1,REPS);
    double h2d_busy = bench_copy(d_b,h_pin_b,BUF,1ULL<<30,cudaMemcpyHostToDevice,s1,REPS);
    CHECK(cudaStreamSynchronize(sk));
    double with_copy = now()-tb0;
    printf("D2H under compute: %.1f GB/s   H2D under compute: %.1f GB/s\n",d2h_busy,h2d_busy);
    printf("kernel runtime: alone %.2fs vs with copies %.2fs (%+.1f%% interference)\n",
           alone, with_copy, (with_copy-alone)/alone*100.0);
  }

  printf("\n-- pageable host memory reference (naive impl), 1GB chunks --\n");
  {
    char* h_page = (char*)malloc(BUF); memset(h_page,5,BUF);
    double d2h = bench_copy(h_page,d_a,BUF,1ULL<<30,cudaMemcpyDeviceToHost,s1,2);
    double h2d = bench_copy(d_b,h_page,BUF,1ULL<<30,cudaMemcpyHostToDevice,s1,2);
    printf("pageable D2H %.1f GB/s   H2D %.1f GB/s\n",d2h,h2d);
    free(h_page);
  }
  printf("\nDONE\n");
  return 0;
}
