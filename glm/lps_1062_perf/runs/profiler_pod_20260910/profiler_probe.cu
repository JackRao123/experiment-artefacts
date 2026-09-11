// Standalone profiler installation check; not a repository test or benchmark.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
    std::exit(1);
  }
}

__global__ void profiler_probe(float* values, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    float value = values[i];
    for (int step = 0; step < 1000; ++step) value = value * 1.000001f + 0.1f;
    values[i] = value;
  }
}

int main() {
  constexpr int count = 1 << 22;
  float* values = nullptr;
  check(cudaMalloc(&values, count * sizeof(float)));
  check(cudaMemset(values, 0, count * sizeof(float)));
  for (int iteration = 0; iteration < 100; ++iteration)
    profiler_probe<<<(count + 255) / 256, 256>>>(values, count);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  check(cudaFree(values));
  std::puts("CUDA profiler probe completed");
}
