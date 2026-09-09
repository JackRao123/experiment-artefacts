SELECT name, COUNT(*) calls, SUM(dur)/1e6 ms,
        AVG(dur)/1e3 avg_us, MIN(dur)/1e3 min_us, MAX(dur)/1e3 max_us
        FROM slice WHERE category='kernel' AND dur>0 GROUP BY name ORDER BY ms DESC;
