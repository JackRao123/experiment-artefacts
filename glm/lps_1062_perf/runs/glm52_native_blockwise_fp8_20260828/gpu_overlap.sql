SELECT (MAX(ts + dur) - MIN(ts)) / 1e6 AS wall_ms,
       SUM(dur) / 1e6 AS gpu_busy_ms,
       100.0 * (1.0 - 1.0 * SUM(dur) / (MAX(ts + dur) - MIN(ts))) AS gpu_idle_pct
FROM slice
WHERE category = 'kernel' AND dur > 0;
