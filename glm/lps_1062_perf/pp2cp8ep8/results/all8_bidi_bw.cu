// all8_bidi_bw.cu -- sustained pinned-host-memory bandwidth bench.
// Extends /tmp/cuda_host_bw.cu (huygens) with: wall-clock sustained duration,
// per-interval stability samples, per-direction (not combined) rates,
// per-GPU NUMA placement policy (local/remote/interleave), and rate limiting.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                  \
                   cudaGetErrorString(error));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

enum class Dir { H2D, D2H };
enum class Numa { Local, Remote, Interleave };

static const int kNode0Cpus[] = {0, 1, 2, 3, 4, 5, 6, 7};
static const int kNode1Cpus[] = {64, 65, 66, 67, 68, 69, 70, 71};

struct Sample { double gbps; };

struct Worker {
  int device;
  Dir dir;
  int cpu;
  int want_node;         // NUMA node the host buffer should land on
  Numa policy;
  size_t bytes;
  int chunk;
  double seconds;
  double interval;
  double target_gbps;    // 0 = unlimited
  std::atomic<int> *ready;
  std::atomic<bool> *start;
  // results
  double elapsed = 0;
  double total_gb = 0;
  int pages_n0 = 0, pages_n1 = 0, pages_unknown = 0;
  std::vector<double> interval_gbps;
  std::vector<double> latencies_ms;
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

static void sample_page_nodes(void *memory, size_t bytes, Worker *w) {
  constexpr int n = 64;
  void *pages[n];
  int status[n];
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  for (int i = 0; i < n; ++i) {
    pages[i] = static_cast<char *>(memory) + (bytes - page) * i / (n - 1);
    status[i] = -1;
  }
  if (syscall(SYS_move_pages, 0, n, pages, nullptr, status, 0) != 0) {
    w->pages_unknown = n;
    return;
  }
  for (int node : status) {
    if (node == 0) ++w->pages_n0;
    else if (node == 1) ++w->pages_n1;
    else ++w->pages_unknown;
  }
}

// Allocate host memory with an explicit NUMA policy, then register it with CUDA
// so it is page-locked (pinned) exactly like cudaMallocHost memory.
static void *alloc_host(size_t bytes, Numa policy, int want_node) {
  if (policy != Numa::Interleave) {
    // First-touch: the calling thread is already pinned to a CPU on want_node,
    // so memset places the pages there.
    void *p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, bytes));
    std::memset(p, 1, bytes);
    return p;
  }
  // MPOL_INTERLEAVE across both nodes, then pin.
  void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
  unsigned long mask = 0x3UL;  // nodes 0 and 1
  const int MPOL_INTERLEAVE = 3;
  if (syscall(SYS_mbind, p, bytes, MPOL_INTERLEAVE, &mask, 3UL, 0) != 0) {
    std::perror("mbind(MPOL_INTERLEAVE)");
    std::exit(1);
  }
  std::memset(p, 1, bytes);
  CUDA_CHECK(cudaHostRegister(p, bytes, cudaHostRegisterDefault));
  return p;
}

static void free_host(void *p, size_t bytes, Numa policy) {
  if (policy != Numa::Interleave) { CUDA_CHECK(cudaFreeHost(p)); return; }
  CUDA_CHECK(cudaHostUnregister(p));
  munmap(p, bytes);
}

using Clock = std::chrono::steady_clock;
static double secs_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

static void run_worker(Worker *w) {
  pin_to_cpu(w->cpu);
  CUDA_CHECK(cudaSetDevice(w->device));

  void *host = alloc_host(w->bytes, w->policy, w->want_node);
  sample_page_nodes(host, w->bytes, w);
  void *dev = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, w->bytes));
  CUDA_CHECK(cudaMemset(dev, 2, w->bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t done;
  CUDA_CHECK(cudaEventCreate(&done));

  const cudaMemcpyKind kind =
      w->dir == Dir::H2D ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
  void *dst = w->dir == Dir::H2D ? dev : host;
  const void *src = w->dir == Dir::H2D ? host : dev;

  // Warm up (also forces any lazy driver mapping work out of the timed region).
  for (int i = 0; i < 4; ++i)
    CUDA_CHECK(cudaMemcpyAsync(dst, src, w->bytes, kind, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  w->ready->fetch_add(1, std::memory_order_release);
  while (!w->start->load(std::memory_order_acquire)) std::this_thread::yield();

  const auto t_start = Clock::now();
  const double gb_per_copy = static_cast<double>(w->bytes) / 1e9;
  double total_gb = 0;
  size_t next_bucket = 0;
  double bucket_gb = 0;
  double bucket_start = 0;
  double paced_next = 0;

  while (secs_since(t_start) < w->seconds) {
    if (w->target_gbps > 0) {
      // Rate-limited: issue one copy per pacing slot, measure completion latency.
      paced_next += gb_per_copy / w->target_gbps;
      double now = secs_since(t_start);
      if (now < paced_next) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(paced_next - now));
      }
      const auto t0 = Clock::now();
      CUDA_CHECK(cudaMemcpyAsync(dst, src, w->bytes, kind, stream));
      CUDA_CHECK(cudaEventRecord(done, stream));
      CUDA_CHECK(cudaEventSynchronize(done));
      w->latencies_ms.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
      total_gb += gb_per_copy;
      bucket_gb += gb_per_copy;
    } else {
      for (int i = 0; i < w->chunk; ++i)
        CUDA_CHECK(cudaMemcpyAsync(dst, src, w->bytes, kind, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      total_gb += gb_per_copy * w->chunk;
      bucket_gb += gb_per_copy * w->chunk;
    }
    const double now = secs_since(t_start);
    if (now >= (next_bucket + 1) * w->interval) {
      w->interval_gbps.push_back(bucket_gb / (now - bucket_start));
      bucket_start = now;
      bucket_gb = 0;
      next_bucket = static_cast<size_t>(now / w->interval);
    }
  }
  w->elapsed = secs_since(t_start);
  w->total_gb = total_gb;

  CUDA_CHECK(cudaEventDestroy(done));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dev));
  free_host(host, w->bytes, w->policy);
}

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5);
  return v[std::min(i, v.size() - 1)];
}

int main(int argc, char **argv) {
  int ngpu = 8;
  std::string mode = "bidir";
  Numa policy = Numa::Local;
  double seconds = 10.0, interval = 1.0;
  size_t buf_mib = 256;
  int chunk = 8;
  double target_d2h = 0, target_h2d = 0;
  std::string label = "run";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() { return a.substr(a.find('=') + 1); };
    if (a.rfind("--gpus=", 0) == 0) ngpu = std::atoi(val().c_str());
    else if (a.rfind("--mode=", 0) == 0) mode = val();
    else if (a.rfind("--numa=", 0) == 0) {
      std::string v = val();
      policy = v == "local" ? Numa::Local
             : v == "remote" ? Numa::Remote
             : v == "interleave" ? Numa::Interleave
             : (std::fprintf(stderr, "bad --numa\n"), std::exit(2), Numa::Local);
    }
    else if (a.rfind("--seconds=", 0) == 0) seconds = std::atof(val().c_str());
    else if (a.rfind("--interval=", 0) == 0) interval = std::atof(val().c_str());
    else if (a.rfind("--buf-mib=", 0) == 0) buf_mib = std::atoll(val().c_str());
    else if (a.rfind("--chunk=", 0) == 0) chunk = std::atoi(val().c_str());
    else if (a.rfind("--target-d2h=", 0) == 0) target_d2h = std::atof(val().c_str());
    else if (a.rfind("--target-h2d=", 0) == 0) target_h2d = std::atof(val().c_str());
    else if (a.rfind("--label=", 0) == 0) label = val();
    else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }

  const size_t bytes = buf_mib * 1024ULL * 1024ULL;
  std::vector<Dir> dirs;
  if (mode == "bidir") dirs = {Dir::D2H, Dir::H2D};
  else if (mode == "d2h") dirs = {Dir::D2H};
  else if (mode == "h2d") dirs = {Dir::H2D};
  else { std::fprintf(stderr, "bad --mode\n"); return 2; }

  std::printf("== %s ==\nmode=%s numa=%s gpus=%d buf=%zuMiB chunk=%d "
              "seconds=%.1f targets(d2h=%.1f h2d=%.1f GB/s)\n",
              label.c_str(), mode.c_str(),
              policy == Numa::Local ? "local" :
              policy == Numa::Remote ? "remote" : "interleave",
              ngpu, buf_mib, chunk, seconds, target_d2h, target_h2d);

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<Worker> workers;
  workers.reserve(ngpu * dirs.size());
  int cpu_slot0 = 0, cpu_slot1 = 0;
  for (int d = 0; d < ngpu; ++d) {
    const int gpu_node = d < 4 ? 0 : 1;
    for (Dir dir : dirs) {
      int want_node;
      if (policy == Numa::Local) want_node = gpu_node;
      else if (policy == Numa::Remote) want_node = 1 - gpu_node;
      else want_node = gpu_node;  // interleave: thread affinity follows GPU
      // The thread runs on a CPU of want_node so first-touch lands there.
      const int cpu = want_node == 0 ? kNode0Cpus[cpu_slot0++ % 8]
                                     : kNode1Cpus[cpu_slot1++ % 8];
      Worker w{};
      w.device = d; w.dir = dir; w.cpu = cpu; w.want_node = want_node;
      w.policy = policy; w.bytes = bytes; w.chunk = chunk;
      w.seconds = seconds; w.interval = interval;
      w.target_gbps = dir == Dir::D2H ? target_d2h : target_h2d;
      w.ready = &ready; w.start = &start;
      workers.push_back(std::move(w));
    }
  }

  std::vector<std::thread> threads;
  for (Worker &w : workers) threads.emplace_back(run_worker, &w);
  while (ready.load(std::memory_order_acquire) != (int)workers.size())
    std::this_thread::yield();
  start.store(true, std::memory_order_release);
  for (std::thread &t : threads) t.join();

  double agg_d2h = 0, agg_h2d = 0;
  std::printf("\n%-5s %-4s %-4s %-18s %10s %10s\n", "gpu", "dir", "cpu",
              "host_pages[N0/N1/?]", "GB/s", "GB moved");
  for (const Worker &w : workers) {
    const double gbps = w.total_gb / w.elapsed;
    if (w.dir == Dir::D2H) agg_d2h += gbps; else agg_h2d += gbps;
    char pages[64];
    std::snprintf(pages, sizeof(pages), "%d/%d/%d", w.pages_n0, w.pages_n1,
                  w.pages_unknown);
    std::printf("%-5d %-4s %-4d %-18s %10.2f %10.1f\n", w.device,
                w.dir == Dir::D2H ? "D2H" : "H2D", w.cpu, pages, gbps,
                w.total_gb);
  }
  const int nd2h = std::count_if(workers.begin(), workers.end(),
                                 [](const Worker &w){ return w.dir == Dir::D2H; });
  const int nh2d = (int)workers.size() - nd2h;
  std::printf("\nAGGREGATE D2H %.2f GB/s (per-gpu mean %.2f)\n", agg_d2h,
              nd2h ? agg_d2h / nd2h : 0.0);
  std::printf("AGGREGATE H2D %.2f GB/s (per-gpu mean %.2f)\n", agg_h2d,
              nh2d ? agg_h2d / nh2d : 0.0);
  std::printf("AGGREGATE BOTH %.2f GB/s\n", agg_d2h + agg_h2d);

  // Per-interval aggregate, to show stability over the run.
  size_t nbuckets = 0;
  for (const Worker &w : workers) nbuckets = std::max(nbuckets, w.interval_gbps.size());
  std::printf("\nstability (aggregate GB/s per %.1fs window):\n", interval);
  std::printf("%-10s %10s %10s\n", "window", "D2H", "H2D");
  for (size_t b = 0; b < nbuckets; ++b) {
    double d = 0, h = 0;
    for (const Worker &w : workers) {
      if (b >= w.interval_gbps.size()) continue;
      if (w.dir == Dir::D2H) d += w.interval_gbps[b]; else h += w.interval_gbps[b];
    }
    std::printf("%-10zu %10.2f %10.2f\n", b, d, h);
  }

  if (target_d2h > 0 || target_h2d > 0) {
    std::printf("\nper-copy completion latency (ms), buf=%zuMiB:\n", buf_mib);
    std::printf("%-5s %-4s %8s %8s %8s %8s %8s\n", "gpu", "dir", "n", "p50",
                "p90", "p99", "max");
    for (const Worker &w : workers) {
      std::printf("%-5d %-4s %8zu %8.3f %8.3f %8.3f %8.3f\n", w.device,
                  w.dir == Dir::D2H ? "D2H" : "H2D", w.latencies_ms.size(),
                  pct(w.latencies_ms, 50), pct(w.latencies_ms, 90),
                  pct(w.latencies_ms, 99), pct(w.latencies_ms, 100));
    }
  }
  return 0;
}
