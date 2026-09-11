
WITH comms AS MATERIALIZED (
 SELECT CAST(EXTRACT_ARG(arg_set_id,'args.External id') AS INT) external_id,
 EXTRACT_ARG(arg_set_id,'args.Process Group Description') pg,
 EXTRACT_ARG(arg_set_id,'args.Collective name') collective
 FROM slice WHERE name='record_param_comms'
)
SELECT k.id,k.ts,k.dur,
 CASE WHEN k.category!='kernel' THEN 'memory'
      WHEN LOWER(k.name) LIKE '%nccl%' THEN 'comm' ELSE 'compute' END kind,
 CASE WHEN LOWER(k.name) LIKE '%nvjet%' OR LOWER(k.name) LIKE '%gemm%'
      OR LOWER(k.name) LIKE '%cublas%' THEN 1 ELSE 0 END gemm,
 COALESCE(c.pg,'') pg,COALESCE(c.collective,'') collective,'' marker
FROM slice k LEFT JOIN comms c
 ON k.category='kernel' AND LOWER(k.name) LIKE '%nccl%'
 AND CAST(EXTRACT_ARG(k.arg_set_id,'args.External id') AS INT)=c.external_id
WHERE k.category IN ('kernel','gpu_memcpy','gpu_memset') AND k.dur>0
UNION ALL
SELECT id,ts,dur,'marker',0,'','',name FROM slice
WHERE category='gpu_user_annotation' AND dur>0
AND (name='CustomFSDP.forward' OR name='ProfilerStep#0'
     OR name GLOB 'block_*/backward_including_recompute')
ORDER BY ts,id
;
