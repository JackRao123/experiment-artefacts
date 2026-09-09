SELECT name,COUNT(*) calls,SUM(dur)/1e6 ms,
        AVG(EXTRACT_ARG(arg_set_id,'args.est. achieved occupancy %')) estimated_occupancy,
        AVG(EXTRACT_ARG(arg_set_id,'args.blocks per SM')) blocks_per_sm
        FROM slice WHERE category='kernel' AND name LIKE '%sdpa%'
        GROUP BY name ORDER BY ms DESC;
