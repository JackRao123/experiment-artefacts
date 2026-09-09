SELECT
  name,
  category,
  COUNT(*) AS calls,
  SUM(dur) / 1e9 AS total_s,
  AVG(dur) / 1e6 AS avg_ms
FROM slice
WHERE dur > 0
GROUP BY name, category
ORDER BY total_s DESC
LIMIT 100;
