SELECT name,
       COUNT(*) AS calls,
       SUM(dur) / 1e6 AS total_ms,
       AVG(dur) / 1e3 AS avg_us
FROM slice
WHERE category IN ('cpu_op', 'user_annotation', 'cuda_runtime')
  AND dur > 0
GROUP BY name
ORDER BY total_ms DESC
LIMIT 40;
