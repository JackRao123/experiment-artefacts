WITH
kernels AS (
  SELECT
    ts,
    ts + dur AS end_ts,
    CASE WHEN
      name GLOB '*hybrid_ep::*'
      OR EXTRACT_ARG(arg_set_id, 'args.Process Group Description') = 'EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP'
    THEN 1 ELSE 0 END AS moe,
    CASE WHEN
      EXTRACT_ARG(arg_set_id, 'args.Process Group Description') IN (
        'CONTEXT_PARALLEL_GROUP',
        'DATA_PARALLEL_GROUP_WITH_CP'
      )
    THEN 1 ELSE 0 END AS cp,
    CASE WHEN
      name NOT GLOB '*hybrid_ep::*'
      AND name NOT GLOB '*ncclDevKernel*'
    THEN 1 ELSE 0 END AS compute
  FROM slice
  WHERE category = 'kernel' AND dur > 0
),
events AS (
  SELECT ts AS t, moe AS d_moe, cp AS d_cp, compute AS d_compute FROM kernels
  UNION ALL
  SELECT end_ts AS t, -moe AS d_moe, -cp AS d_cp, -compute AS d_compute FROM kernels
),
grouped AS (
  SELECT t, SUM(d_moe) AS d_moe, SUM(d_cp) AS d_cp, SUM(d_compute) AS d_compute
  FROM events
  GROUP BY t
),
active AS (
  SELECT
    t,
    LEAD(t) OVER (ORDER BY t) AS next_t,
    SUM(d_moe) OVER (ORDER BY t ROWS UNBOUNDED PRECEDING) AS moe,
    SUM(d_cp) OVER (ORDER BY t ROWS UNBOUNDED PRECEDING) AS cp,
    SUM(d_compute) OVER (ORDER BY t ROWS UNBOUNDED PRECEDING) AS compute
  FROM grouped
)
SELECT
  SUM(CASE WHEN moe > 0 THEN next_t - t ELSE 0 END) / 1e9 AS moe_union_s,
  SUM(CASE WHEN moe > 0 AND compute > 0 THEN next_t - t ELSE 0 END) / 1e9 AS moe_compute_overlap_s,
  SUM(CASE WHEN moe > 0 AND compute = 0 THEN next_t - t ELSE 0 END) / 1e9 AS moe_exposed_s,
  SUM(CASE WHEN cp > 0 THEN next_t - t ELSE 0 END) / 1e9 AS cp_union_s,
  SUM(CASE WHEN cp > 0 AND compute > 0 THEN next_t - t ELSE 0 END) / 1e9 AS cp_compute_overlap_s,
  SUM(CASE WHEN cp > 0 AND compute = 0 THEN next_t - t ELSE 0 END) / 1e9 AS cp_exposed_s,
  SUM(CASE WHEN compute > 0 THEN next_t - t ELSE 0 END) / 1e9 AS compute_union_s,
  SUM(CASE WHEN moe > 0 OR cp > 0 OR compute > 0 THEN next_t - t ELSE 0 END) / 1e9 AS gpu_active_union_s
FROM active
WHERE next_t IS NOT NULL;
