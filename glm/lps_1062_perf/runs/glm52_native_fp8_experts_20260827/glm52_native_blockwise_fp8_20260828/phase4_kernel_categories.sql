SELECT CASE
         WHEN name LIKE '%hybrid_ep%' THEN 'HybridEP dispatch/combine/sync'
         WHEN name LIKE '%sparse_attention%' OR name LIKE '%dsa_%' OR name LIKE '%indexer%' THEN 'DSA attention/indexer'
         WHEN name LIKE 'nvjet%' OR name LIKE '%gemm%' THEN 'GEMM'
         WHEN name LIKE 'ncclDevKernel%' THEN 'NCCL'
         WHEN name LIKE '%permute_kernel%' OR name LIKE '%unpermute_kernel%' OR name LIKE '%CatArray%' THEN 'Permutation/concatenation'
         WHEN name LIKE '%direct_copy%' OR name LIKE '%bfloat16_copy%' OR name LIKE '%MulFunctor<float>%' THEN 'FP8 dequant/cast candidates'
         ELSE 'Other'
       END AS kernel_class,
       COUNT(*) AS calls,
       SUM(dur) / 1e6 AS total_ms
FROM slice
WHERE category = 'kernel' AND dur > 0
GROUP BY kernel_class
ORDER BY total_ms DESC;
