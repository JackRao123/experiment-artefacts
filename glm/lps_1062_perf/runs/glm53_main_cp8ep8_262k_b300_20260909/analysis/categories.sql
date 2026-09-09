SELECT
  category,
  COUNT(*) AS calls,
  SUM(dur) / 1e9 AS total_s
FROM slice
WHERE dur > 0
GROUP BY category
ORDER BY total_s DESC
LIMIT 30;
