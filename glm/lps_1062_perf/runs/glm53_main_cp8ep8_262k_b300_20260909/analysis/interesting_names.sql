SELECT
  category,
  name,
  COUNT(*) AS calls,
  SUM(dur) / 1e9 AS total_s,
  AVG(dur) / 1e6 AS avg_ms
FROM slice
WHERE dur > 0
  AND (
    LOWER(name) GLOB '*layer*'
    OR LOWER(name) GLOB '*forward*'
    OR LOWER(name) GLOB '*backward*'
    OR LOWER(name) GLOB '*recompute*'
    OR LOWER(name) GLOB '*dispatch*'
    OR LOWER(name) GLOB '*combine*'
    OR LOWER(name) GLOB '*attention*'
    OR LOWER(name) GLOB '*alltoall*'
    OR LOWER(name) GLOB '*all_to_all*'
  )
GROUP BY category, name
ORDER BY total_s DESC
LIMIT 250;
