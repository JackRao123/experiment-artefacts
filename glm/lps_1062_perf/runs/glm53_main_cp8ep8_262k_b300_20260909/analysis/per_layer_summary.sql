WITH RECURSIVE
layers AS (
  SELECT
    id,
    name,
    ts,
    dur,
    CASE
      WHEN name = 'CheckpointFunction' THEN ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1
      ELSE 78 - ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts)
    END AS layer_number
  FROM slice
  WHERE name IN ('CheckpointFunction', 'CheckpointFunctionBackward')
),
descendants(layer_id, slice_id, name, ts, depth) AS (
  SELECT l.id, s.id, s.name, s.ts, 1
  FROM layers l
  JOIN slice s ON s.parent_id = l.id
  UNION ALL
  SELECT d.layer_id, s.id, s.name, s.ts, d.depth + 1
  FROM descendants d
  JOIN slice s ON s.parent_id = d.slice_id
  WHERE d.depth < 30
),
actual_backward_start AS (
  SELECT layer_id, MIN(ts) AS ts
  FROM descendants
  WHERE name GLOB 'autograd::engine::evaluate_function:*'
  GROUP BY layer_id
),
metrics AS (
  SELECT
    CASE WHEN l.layer_number < 3 THEN 'dense' ELSE 'moe' END AS layer_type,
    CASE
      WHEN l.name = 'CheckpointFunction' THEN 'forward'
      ELSE 'backward envelope'
    END AS phase,
    l.dur / 1e6 AS envelope_ms,
    CASE WHEN l.name = 'CheckpointFunctionBackward' THEN (a.ts - l.ts) / 1e6 END AS recompute_ms,
    CASE WHEN l.name = 'CheckpointFunctionBackward' THEN (l.ts + l.dur - a.ts) / 1e6 END AS actual_backward_ms
  FROM layers l
  LEFT JOIN actual_backward_start a ON a.layer_id = l.id
)
SELECT
  layer_type,
  phase,
  COUNT(*) AS layers,
  AVG(envelope_ms) AS avg_envelope_ms,
  MIN(envelope_ms) AS min_envelope_ms,
  MAX(envelope_ms) AS max_envelope_ms,
  AVG(recompute_ms) AS avg_recompute_ms,
  AVG(actual_backward_ms) AS avg_actual_backward_ms
FROM metrics
GROUP BY layer_type, phase
ORDER BY phase, layer_type;
