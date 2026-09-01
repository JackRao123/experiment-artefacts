SELECT name, COUNT(*) AS calls, SUM(dur)/1e6 AS total_ms,
       SUM(dur)/1e3/COUNT(*) AS avg_us
FROM slice
WHERE dur > 0 AND category = 'kernel'
  AND (name LIKE '%sgemm%' OR name LIKE '%gemm%' OR name LIKE '%GEMM%')
GROUP BY name
ORDER BY total_ms DESC
LIMIT 80;
