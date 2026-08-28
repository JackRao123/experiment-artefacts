SELECT name,
       COUNT(*) AS calls,
       SUM(dur) / 1e6 AS total_ms,
       SUM(dur) / 1e3 / COUNT(*) AS avg_us
FROM slice
WHERE dur > 0 AND category = 'kernel'
GROUP BY name
ORDER BY total_ms DESC
LIMIT 40;
