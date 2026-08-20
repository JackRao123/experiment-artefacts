#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <string>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Measures pinned host-memory copies with verified NUMA page placement.

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error = (call);                                                  \
    if (error != cudaSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(error));                                   \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

enum class Direction { H2D, D2H, Bidir };

struct Worker {
  int device;
  int cpu;
  Direction direction;
  size_t bytes;
  int iterations;
  std::atomic<int> *ready;
  std::atomic<bool> *start;
  double seconds = 0;
  int host_pages_node0 = 0;
  int host_pages_node1 = 0;
  int host_pages_unknown = 0;
};

static void pin_to_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    std::perror("sched_setaffinity");
    std::exit(1);
  }
}

static void sample_page_nodes(void *memory, size_t bytes, Worker *worker) {
  constexpr int sample_count = 64;
  void *pages[sample_count];
  int status[sample_count];
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  for (int i = 0; i < sample_count; ++i) {
    const size_t offset = (bytes - page_size) * i / (sample_count - 1);
    pages[i] = static_cast<char *>(memory) + offset;
    status[i] = -1;
  }
  const long result = syscall(SYS_move_pages, 0, sample_count, pages, nullptr,
                              status, 0);
  if (result != 0) {
    worker->host_pages_unknown = sample_count;
    return;
  }
  for (int node : status) {
    if (node == 0) {
      ++worker->host_pages_node0;
    } else if (node == 1) {
      ++worker->host_pages_node1;
    } else {
      ++worker->host_pages_unknown;
    }
  }
}

static void run_worker(Worker *worker) {
  pin_to_cpu(worker->cpu);
  CUDA_CHECK(cudaSetDevice(worker->device));

  void *host_a = nullptr;
  void *host_b = nullptr;
  void *device_a = nullptr;
  void *device_b = nullptr;
  CUDA_CHECK(cudaMallocHost(&host_a, worker->bytes));
  CUDA_CHECK(cudaMalloc(&device_a, worker->bytes));
  std::memset(host_a, 1, worker->bytes);
  sample_page_nodes(host_a, worker->bytes, worker);
  CUDA_CHECK(cudaMemset(device_a, 1, worker->bytes));

  cudaStream_t stream_a;
  cudaStream_t stream_b;
  CUDA_CHECK(cudaStreamCreate(&stream_a));
  CUDA_CHECK(cudaStreamCreate(&stream_b));

  if (worker->direction == Direction::Bidir) {
    CUDA_CHECK(cudaMallocHost(&host_b, worker->bytes));
    CUDA_CHECK(cudaMalloc(&device_b, worker->bytes));
    std::memset(host_b, 2, worker->bytes);
    CUDA_CHECK(cudaMemset(device_b, 2, worker->bytes));
  }

  CUDA_CHECK(cudaMemcpyAsync(device_a, host_a, worker->bytes,
                             cudaMemcpyHostToDevice, stream_a));
  CUDA_CHECK(cudaMemcpyAsync(host_a, device_a, worker->bytes,
                             cudaMemcpyDeviceToHost, stream_a));
  CUDA_CHECK(cudaDeviceSynchronize());

  worker->ready->fetch_add(1, std::memory_order_release);
  while (!worker->start->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < worker->iterations; ++i) {
    if (worker->direction == Direction::H2D) {
      CUDA_CHECK(cudaMemcpyAsync(device_a, host_a, worker->bytes,
                                 cudaMemcpyHostToDevice, stream_a));
    } else if (worker->direction == Direction::D2H) {
      CUDA_CHECK(cudaMemcpyAsync(host_a, device_a, worker->bytes,
                                 cudaMemcpyDeviceToHost, stream_a));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(device_a, host_a, worker->bytes,
                                 cudaMemcpyHostToDevice, stream_a));
      CUDA_CHECK(cudaMemcpyAsync(host_b, device_b, worker->bytes,
                                 cudaMemcpyDeviceToHost, stream_b));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto end = std::chrono::steady_clock::now();
  worker->seconds = std::chrono::duration<double>(end - begin).count();

  CUDA_CHECK(cudaStreamDestroy(stream_a));
  CUDA_CHECK(cudaStreamDestroy(stream_b));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFreeHost(host_a));
  if (worker->direction == Direction::Bidir) {
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFreeHost(host_b));
  }
}

static Direction parse_direction(const char *value) {
  const std::string direction(value);
  if (direction == "h2d") return Direction::H2D;
  if (direction == "d2h") return Direction::D2H;
  if (direction == "bidir") return Direction::Bidir;
  std::fprintf(stderr, "direction must be h2d, d2h, or bidir\n");
  std::exit(2);
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr,
                 "usage: %s <device|all> <h2d|d2h|bidir> [alloc-cpu]\n",
                 argv[0]);
    return 2;
  }

  const Direction direction = parse_direction(argv[2]);
  const size_t bytes = 256ULL * 1024 * 1024;
  const int iterations = 40;
  const int local_cpus[] = {0, 1, 2, 3, 64, 65, 66, 67};
  const int worker_count = std::string(argv[1]) == "all" ? 8 : 1;
  const int first_device = worker_count == 1 ? std::atoi(argv[1]) : 0;

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<Worker> workers;
  std::vector<std::thread> threads;
  workers.reserve(worker_count);
  threads.reserve(worker_count);

  for (int i = 0; i < worker_count; ++i) {
    const int device = first_device + i;
    const int cpu = argc == 4 ? std::atoi(argv[3]) : local_cpus[device];
    workers.push_back(
        {device, cpu, direction, bytes, iterations, &ready, &start});
  }
  for (Worker &worker : workers) threads.emplace_back(run_worker, &worker);
  while (ready.load(std::memory_order_acquire) != worker_count) {
    std::this_thread::yield();
  }

  start.store(true, std::memory_order_release);
  for (std::thread &thread : threads) thread.join();
  const double directions = direction == Direction::Bidir ? 2.0 : 1.0;
  const double total_gb = directions * worker_count * iterations * bytes / 1e9;
  double max_seconds = 0;

  for (const Worker &worker : workers) {
    if (worker.seconds > max_seconds) max_seconds = worker.seconds;
    const double worker_gb = directions * iterations * bytes / 1e9;
    std::printf("gpu%d cpu%d pages[N0=%d,N1=%d,unknown=%d] %.2f GB/s aggregate",
                worker.device, worker.cpu, worker.host_pages_node0,
                worker.host_pages_node1, worker.host_pages_unknown,
                worker_gb / worker.seconds);
    if (direction == Direction::Bidir) {
      std::printf(" (%.2f GB/s per direction)",
                  worker_gb / worker.seconds / 2.0);
    }
    std::printf("\n");
  }
  std::printf("all-gpu aggregate %.2f GB/s", total_gb / max_seconds);
  if (direction == Direction::Bidir) {
    std::printf(" (%.2f GB/s per direction)", total_gb / max_seconds / 2.0);
  }
  std::printf("\n");
  return 0;
}
