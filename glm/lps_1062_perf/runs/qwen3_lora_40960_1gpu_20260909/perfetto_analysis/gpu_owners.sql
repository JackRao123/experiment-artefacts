WITH owners AS MATERIALIZED
        (SELECT id,name,EXTRACT_ARG(arg_set_id,'args.External id') ext FROM slice WHERE category='cpu_op')
        SELECT o.name,COUNT(*) kernels,SUM(k.dur)/1e6 gpu_ms FROM slice k
        JOIN owners o ON o.ext=EXTRACT_ARG(k.arg_set_id,'args.External id')
        WHERE k.category='kernel' GROUP BY o.name ORDER BY gpu_ms DESC LIMIT 70;
