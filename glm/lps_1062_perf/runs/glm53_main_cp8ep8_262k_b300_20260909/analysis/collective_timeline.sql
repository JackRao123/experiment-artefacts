WITH bounds AS (
  SELECT MIN(ts) AS trace_start FROM slice
)
SELECT
  (s.ts - trace_start) / 1e6 AS start_ms,
  s.dur / 1e6 AS dur_ms,
  t.name AS track_name,
  CASE
    WHEN s.name GLOB '*hybrid_ep::device_sync_kernel*' THEN 'HybridEP sync'
    WHEN s.name GLOB '*hybrid_ep::dispatch_kernel*' THEN 'HybridEP dispatch'
    WHEN s.name GLOB '*hybrid_ep::combine_kernel*' THEN 'HybridEP combine'
    WHEN s.name GLOB '*ncclDevKernel_ReduceScatter*' THEN 'NCCL reduce-scatter'
    WHEN s.name GLOB '*ncclDevKernel_AllGather*' THEN 'NCCL all-gather'
    WHEN s.name GLOB '*ncclDevKernel_AllReduce*' THEN 'NCCL all-reduce'
  END AS class,
  s.name
FROM slice s
JOIN track t ON t.id = s.track_id
CROSS JOIN bounds
WHERE s.category = 'kernel'
  AND (
    s.name GLOB '*hybrid_ep::device_sync_kernel*'
    OR s.name GLOB '*hybrid_ep::dispatch_kernel*'
    OR s.name GLOB '*hybrid_ep::combine_kernel*'
    OR s.name GLOB '*ncclDevKernel_ReduceScatter*'
    OR s.name GLOB '*ncclDevKernel_AllGather*'
    OR s.name GLOB '*ncclDevKernel_AllReduce*'
  )
ORDER BY s.ts;
