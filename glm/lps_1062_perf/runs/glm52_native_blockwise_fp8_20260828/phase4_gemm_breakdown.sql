SELECT CASE
         WHEN name LIKE 'nvjet%' THEN 'NVJet BF16 GEMM'
         WHEN name LIKE 'cutlass3x_sm100_simt_sgemm%' THEN 'FP32 output-head GEMM'
         ELSE 'Other GEMM'
       END AS gemm_class,
       COUNT(*) AS calls,
       SUM(dur) / 1e6 AS total_ms,
       AVG(dur) / 1e3 AS avg_us
FROM slice
WHERE category = 'kernel'
  AND dur > 0
  AND (name LIKE 'nvjet%' OR name LIKE '%gemm%')
GROUP BY gemm_class
ORDER BY total_ms DESC;
