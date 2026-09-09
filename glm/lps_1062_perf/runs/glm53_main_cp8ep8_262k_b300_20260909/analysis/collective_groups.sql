SELECT
  CASE
    WHEN name GLOB '*AllGather*' THEN 'all-gather'
    WHEN name GLOB '*ReduceScatter*' THEN 'reduce-scatter'
    WHEN name GLOB '*AllReduce*' THEN 'all-reduce'
    ELSE name
  END AS collective,
  EXTRACT_ARG(arg_set_id, 'args.Process Group Description') AS process_group,
  EXTRACT_ARG(arg_set_id, 'args.dtype') AS dtype,
  EXTRACT_ARG(arg_set_id, 'args.In msg nelems') AS input_elements,
  COUNT(*) AS calls,
  SUM(dur) / 1e9 AS summed_s,
  AVG(dur) / 1e6 AS avg_ms,
  MAX(dur) / 1e6 AS max_ms
FROM slice
WHERE category = 'kernel' AND name GLOB 'ncclDevKernel*'
GROUP BY collective, process_group, dtype, input_elements
ORDER BY summed_s DESC;
