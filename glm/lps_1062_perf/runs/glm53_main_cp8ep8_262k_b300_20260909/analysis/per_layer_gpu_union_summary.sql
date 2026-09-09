WITH RECURSIVE
layers AS (
  SELECT
    id,
    name,
    CASE
      WHEN name = 'CheckpointFunction' THEN ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1
      ELSE 78 - ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts)
    END AS layer_number
  FROM slice
  WHERE name IN ('CheckpointFunction', 'CheckpointFunctionBackward')
),
kernel_launch AS (
  SELECT k.id AS kernel_id, k.ts, k.ts + k.dur AS end_ts, launch.parent_id AS ancestor_id
  FROM flow f
  JOIN slice launch ON launch.id = f.slice_out
  JOIN slice k ON k.id = f.slice_in
  WHERE k.category = 'kernel' AND k.dur > 0
),
ancestors(kernel_id, slice_id, depth) AS (
  SELECT kernel_id, ancestor_id, 1 FROM kernel_launch WHERE ancestor_id IS NOT NULL
  UNION ALL
  SELECT a.kernel_id, s.parent_id, a.depth + 1
  FROM ancestors a
  JOIN slice s ON s.id = a.slice_id
  WHERE s.parent_id IS NOT NULL AND a.depth < 30
),
kernel_layer AS (
  SELECT DISTINCT
    k.kernel_id,
    k.ts,
    k.end_ts,
    l.id AS layer_id,
    l.name AS envelope,
    l.layer_number
  FROM kernel_launch k
  JOIN ancestors a ON a.kernel_id = k.kernel_id
  JOIN layers l ON l.id = a.slice_id
),
semantic_backward AS (
  SELECT DISTINCT a.kernel_id
  FROM ancestors a
  JOIN slice s ON s.id = a.slice_id
  WHERE s.name GLOB '*Backward*'
    AND s.name NOT IN ('CheckpointFunctionBackward', 'autograd::engine::evaluate_function: CheckpointFunctionBackward')
),
classified AS (
  SELECT
    kl.layer_id,
    kl.layer_number,
    CASE WHEN kl.layer_number < 3 THEN 'dense' ELSE 'moe' END AS layer_type,
    CASE
      WHEN kl.envelope = 'CheckpointFunction' THEN 'forward'
      WHEN sb.kernel_id IS NOT NULL THEN 'actual_backward'
      ELSE 'recompute_forward'
    END AS phase,
    kl.ts,
    kl.end_ts
  FROM kernel_layer kl
  LEFT JOIN semantic_backward sb ON sb.kernel_id = kl.kernel_id
),
events AS (
  SELECT layer_id, layer_number, layer_type, phase, ts AS t, 1 AS delta FROM classified
  UNION ALL
  SELECT layer_id, layer_number, layer_type, phase, end_ts AS t, -1 AS delta FROM classified
),
grouped AS (
  SELECT layer_id, layer_number, layer_type, phase, t, SUM(delta) AS delta
  FROM events
  GROUP BY layer_id, layer_number, layer_type, phase, t
),
active AS (
  SELECT
    layer_id,
    layer_number,
    layer_type,
    phase,
    t,
    LEAD(t) OVER (PARTITION BY layer_id, phase ORDER BY t) AS next_t,
    SUM(delta) OVER (PARTITION BY layer_id, phase ORDER BY t ROWS UNBOUNDED PRECEDING) AS active
  FROM grouped
),
per_layer AS (
  SELECT
    layer_id,
    layer_number,
    layer_type,
    phase,
    SUM(CASE WHEN active > 0 THEN next_t - t ELSE 0 END) / 1e6 AS gpu_union_ms
  FROM active
  WHERE next_t IS NOT NULL
  GROUP BY layer_id, layer_number, layer_type, phase
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY layer_type, phase ORDER BY gpu_union_ms) AS rn,
    COUNT(*) OVER (PARTITION BY layer_type, phase) AS n
  FROM per_layer
)
SELECT
  layer_type,
  phase,
  COUNT(*) AS layers,
  AVG(gpu_union_ms) AS avg_gpu_union_ms,
  MAX(CASE WHEN rn = (n + 1) / 2 THEN gpu_union_ms END) AS median_gpu_union_ms,
  MIN(gpu_union_ms) AS min_gpu_union_ms,
  MAX(gpu_union_ms) AS max_gpu_union_ms
FROM ranked
GROUP BY layer_type, phase
ORDER BY layer_type, phase;
