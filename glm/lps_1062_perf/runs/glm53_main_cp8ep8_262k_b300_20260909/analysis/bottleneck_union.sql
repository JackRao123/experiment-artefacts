WITH
classified AS (
  SELECT
    ts,
    ts + dur AS end_ts,
    CASE
      WHEN name GLOB '*hybrid_ep::*' OR EXTRACT_ARG(arg_set_id, 'args.Process Group Description') = 'EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP' THEN '1 MoE communication'
      WHEN EXTRACT_ARG(arg_set_id, 'args.Process Group Description') IN ('CONTEXT_PARALLEL_GROUP', 'DATA_PARALLEL_GROUP_WITH_CP') THEN '5 CP communication'
      WHEN name GLOB '*sparse_attention*' OR name GLOB '*sparse_attn*' OR name GLOB '*indexer_forward*' OR name GLOB '*indexer_topk*' THEN '2 DSA attention compute'
      WHEN name GLOB '*nvjet*' THEN '3 GEMM compute'
      WHEN name GLOB '*permute*' THEN '6 MoE local permutation'
      ELSE '4 Other compute'
    END AS class
  FROM slice
  WHERE category = 'kernel' AND dur > 0
),
events AS (
  SELECT class, ts AS t, 1 AS delta FROM classified
  UNION ALL
  SELECT class, end_ts AS t, -1 AS delta FROM classified
),
grouped AS (
  SELECT class, t, SUM(delta) AS delta
  FROM events
  GROUP BY class, t
),
active AS (
  SELECT
    class,
    t,
    LEAD(t) OVER (PARTITION BY class ORDER BY t) AS next_t,
    SUM(delta) OVER (PARTITION BY class ORDER BY t ROWS UNBOUNDED PRECEDING) AS active
  FROM grouped
)
SELECT
  class,
  SUM(CASE WHEN active > 0 THEN next_t - t ELSE 0 END) / 1e9 AS union_s
FROM active
WHERE next_t IS NOT NULL
GROUP BY class
ORDER BY union_s DESC;
