WITH RECURSIVE
layers AS (
  SELECT
    id,
    name,
    ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1 AS ordinal,
    CASE
      WHEN name = 'CheckpointFunction' THEN ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1
      ELSE 78 - ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts)
    END AS layer_number
  FROM slice
  WHERE name IN ('CheckpointFunction', 'CheckpointFunctionBackward')
),
kernel_launch AS (
  SELECT k.id AS kernel_id, k.name AS kernel_name, k.dur, launch.parent_id AS ancestor_id
  FROM flow f
  JOIN slice launch ON launch.id = f.slice_out
  JOIN slice k ON k.id = f.slice_in
  WHERE k.category = 'kernel' AND k.dur > 0
),
ancestors(kernel_id, kernel_name, dur, slice_id, depth) AS (
  SELECT kernel_id, kernel_name, dur, ancestor_id, 1
  FROM kernel_launch
  WHERE ancestor_id IS NOT NULL
  UNION ALL
  SELECT a.kernel_id, a.kernel_name, a.dur, s.parent_id, a.depth + 1
  FROM ancestors a
  JOIN slice s ON s.id = a.slice_id
  WHERE s.parent_id IS NOT NULL AND a.depth < 30
),
kernel_layer AS (
  SELECT DISTINCT
    k.kernel_id,
    k.kernel_name,
    k.dur,
    l.id AS layer_id,
    l.name AS envelope,
    l.layer_number
  FROM kernel_launch k
  JOIN ancestors a ON a.kernel_id = k.kernel_id
  JOIN layers l ON l.id = a.slice_id
),
semantic_backward AS (
  SELECT DISTINCT a.kernel_id
  FROM ancestors a
  JOIN slice s ON s.id = a.slice_id
  WHERE s.name GLOB '*Backward*'
    AND s.name NOT IN ('CheckpointFunctionBackward', 'autograd::engine::evaluate_function: CheckpointFunctionBackward')
),
classified AS (
  SELECT
    CASE WHEN kl.layer_number < 3 THEN 'dense' ELSE 'moe' END AS layer_type,
    CASE
      WHEN kl.envelope = 'CheckpointFunction' THEN 'forward'
      WHEN sb.kernel_id IS NOT NULL THEN 'actual_backward'
      ELSE 'recompute_forward'
    END AS phase,
    CASE
      WHEN kl.kernel_name GLOB '*hybrid_ep::device_sync_kernel*' THEN 'HybridEP sync'
      WHEN kl.kernel_name GLOB '*hybrid_ep::dispatch_kernel*' THEN 'HybridEP dispatch'
      WHEN kl.kernel_name GLOB '*hybrid_ep::combine_kernel*' THEN 'HybridEP combine'
      WHEN kl.kernel_name GLOB '*hybrid_ep::*' THEN 'HybridEP metadata'
      WHEN kl.kernel_name GLOB '*AllReduce_Sum_u64*' THEN 'HybridEP metadata all-reduce'
      WHEN kl.kernel_name GLOB '*ncclDevKernel_ReduceScatter*' THEN 'CP reduce-scatter'
      WHEN kl.kernel_name GLOB '*ncclDevKernel_AllGather*' THEN 'CP all-gather'
      WHEN kl.kernel_name GLOB '*AllReduce_Sum_f32*' THEN 'CP/grad all-reduce'
      WHEN kl.kernel_name GLOB '*sparse_attention_backward*' THEN 'DSA backward'
      WHEN kl.kernel_name GLOB '*sparse_attn_fwd_kernel*' THEN 'DSA forward'
      WHEN kl.kernel_name GLOB '*indexer_forward*' THEN 'DSA indexer forward'
      WHEN kl.kernel_name GLOB '*indexer_topk*' THEN 'DSA indexer top-k'
      WHEN kl.kernel_name GLOB '*permute_kernel*' OR kl.kernel_name GLOB '*permute_preprocessing_kernel*' THEN 'MoE local permute'
      WHEN kl.kernel_name GLOB '*unpermute_kernel*' THEN 'MoE local unpermute'
      WHEN kl.kernel_name GLOB '*nvjet*' THEN 'GEMM (nvjet)'
      ELSE 'other kernels'
    END AS kernel_class,
    kl.dur
  FROM kernel_layer kl
  LEFT JOIN semantic_backward sb ON sb.kernel_id = kl.kernel_id
)
SELECT
  layer_type,
  phase,
  kernel_class,
  COUNT(*) AS calls,
  SUM(dur) / 1e9 AS summed_s,
  AVG(dur) / 1e6 AS avg_ms
FROM classified
GROUP BY layer_type, phase, kernel_class
ORDER BY layer_type, phase, summed_s DESC;
