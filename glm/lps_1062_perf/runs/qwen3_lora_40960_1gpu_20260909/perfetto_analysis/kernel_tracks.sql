SELECT track_id,COUNT(*) kernels,SUM(dur)/1e6 ms
        FROM slice WHERE category='kernel' GROUP BY track_id ORDER BY ms DESC;
