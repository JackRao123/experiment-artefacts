SELECT
  CASE
    WHEN name GLOB '*hybrid_ep::device_sync_kernel*' THEN 'HybridEP device sync'
    WHEN name GLOB '*hybrid_ep::dispatch_kernel*' THEN 'HybridEP dispatch'
    WHEN name GLOB '*hybrid_ep::combine_kernel*' THEN 'HybridEP combine'
    WHEN name GLOB '*ncclDevKernel_ReduceScatter*' THEN 'CP reduce-scatter'
    WHEN name GLOB '*ncclDevKernel_AllGather*' THEN 'NCCL all-gather'
    WHEN name GLOB '*ncclDevKernel_AllReduce*' THEN 'NCCL all-reduce'
    WHEN name GLOB '*sparse_attention_backward*' THEN 'DSA backward'
    WHEN name GLOB '*sparse_attn_fwd_kernel*' THEN 'DSA forward'
    WHEN name GLOB '*indexer_forward*' THEN 'DSA indexer forward'
    WHEN name GLOB '*indexer_topk*' THEN 'DSA indexer top-k'
    WHEN name GLOB '*permute_kernel*' OR name GLOB '*permute_preprocessing_kernel*' THEN 'MoE local permute'
    WHEN name GLOB '*unpermute_kernel*' THEN 'MoE local unpermute'
    WHEN name GLOB '*nvjet*' THEN 'GEMM (nvjet)'
    ELSE 'other kernels'
  END AS kernel_class,
  COUNT(*) AS calls,
  SUM(dur) / 1e9 AS summed_s,
  AVG(dur) / 1e6 AS avg_ms
FROM slice
WHERE category = 'kernel' AND dur > 0
GROUP BY kernel_class
ORDER BY summed_s DESC;
