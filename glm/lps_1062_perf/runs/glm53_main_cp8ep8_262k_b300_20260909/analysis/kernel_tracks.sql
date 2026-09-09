SELECT
  t.id AS track_id,
  t.name AS track_name,
  COUNT(*) AS kernels,
  SUM(s.dur) / 1e9 AS summed_s,
  MIN(s.ts) / 1e9 AS first_s,
  MAX(s.ts + s.dur) / 1e9 AS last_s
FROM slice s
JOIN track t ON t.id = s.track_id
WHERE s.category = 'kernel' AND s.dur > 0
GROUP BY t.id, t.name
ORDER BY summed_s DESC;
