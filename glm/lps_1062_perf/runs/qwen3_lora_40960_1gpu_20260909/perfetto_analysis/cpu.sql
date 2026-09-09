SELECT category,name,COUNT(*) calls,SUM(dur)/1e6 inclusive_ms,
        MAX(dur)/1e6 max_ms FROM slice
        WHERE category IN ('cpu_op','cuda_runtime','user_annotation') AND dur>0
        GROUP BY category,name ORDER BY inclusive_ms DESC LIMIT 100;
