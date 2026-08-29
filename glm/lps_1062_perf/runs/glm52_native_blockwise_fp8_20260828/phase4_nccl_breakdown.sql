WITH nccl_kernels AS (
  SELECT name, dur
  FROM slice
  WHERE name LIKE 'ncclDevKernel%' AND dur > 0
)
SELECT CASE
         WHEN name LIKE '%SendRecv%' THEN 'EP SendRecv'
         WHEN name LIKE '%AllGather%' THEN 'AllGather'
         WHEN name LIKE '%ReduceScatter%' THEN 'ReduceScatter'
         WHEN name LIKE '%AllReduce%' THEN 'AllReduce'
         ELSE name
       END AS collective_class,
       COUNT(*) AS calls,
       SUM(dur) / 1e6 AS total_ms,
       AVG(dur) / 1e3 AS avg_us,
       MAX(dur) / 1e3 AS max_us
FROM nccl_kernels
GROUP BY collective_class
ORDER BY total_ms DESC;
