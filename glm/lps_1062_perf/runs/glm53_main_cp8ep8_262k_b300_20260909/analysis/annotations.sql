SELECT
  category,
  name,
  COUNT(*) AS calls,
  SUM(dur) / 1e9 AS total_s,
  AVG(dur) / 1e6 AS avg_ms,
  MIN(dur) / 1e6 AS min_ms,
  MAX(dur) / 1e6 AS max_ms
FROM slice
WHERE dur > 0
  AND category IN ('user_annotation', 'gpu_user_annotation')
GROUP BY category, name
ORDER BY category, total_s DESC;
