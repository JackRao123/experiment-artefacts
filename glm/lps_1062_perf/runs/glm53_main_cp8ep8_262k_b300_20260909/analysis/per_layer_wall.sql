WITH RECURSIVE
layers AS (
  SELECT
    id,
    name,
    ts,
    dur,
    ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1 AS ordinal,
    CASE
      WHEN name = 'CheckpointFunction' THEN ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1
      ELSE 78 - ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts)
    END AS layer_number
  FROM slice
  WHERE name IN ('CheckpointFunction', 'CheckpointFunctionBackward')
),
descendants(layer_id, slice_id, name, category, ts, depth) AS (
  SELECT l.id, s.id, s.name, s.category, s.ts, 1
  FROM layers l
  JOIN slice s ON s.parent_id = l.id
  UNION ALL
  SELECT d.layer_id, s.id, s.name, s.category, s.ts, d.depth + 1
  FROM descendants d
  JOIN slice s ON s.parent_id = d.slice_id
  WHERE d.depth < 30
),
actual_backward_start AS (
  SELECT
    layer_id,
    MIN(ts) AS ts
  FROM descendants
  WHERE name GLOB 'autograd::engine::evaluate_function:*'
  GROUP BY layer_id
)
SELECT
  l.name AS envelope,
  l.ordinal,
  l.layer_number,
  CASE WHEN l.layer_number < 3 THEN 'dense' ELSE 'moe' END AS layer_type,
  l.dur / 1e6 AS envelope_ms,
  CASE
    WHEN l.name = 'CheckpointFunctionBackward' THEN (a.ts - l.ts) / 1e6
  END AS recompute_ms,
  CASE
    WHEN l.name = 'CheckpointFunctionBackward' THEN (l.ts + l.dur - a.ts) / 1e6
  END AS actual_backward_ms
FROM layers l
LEFT JOIN actual_backward_start a ON a.layer_id = l.id
ORDER BY l.ts;
