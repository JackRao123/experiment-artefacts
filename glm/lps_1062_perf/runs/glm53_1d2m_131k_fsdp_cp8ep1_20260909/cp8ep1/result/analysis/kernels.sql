SELECT name,COUNT(*) calls,SUM(dur)/1e6 sum_ms,MAX(dur)/1e6 max_ms FROM slice WHERE category='kernel' AND dur>0 GROUP BY name ORDER BY sum_ms DESC;
