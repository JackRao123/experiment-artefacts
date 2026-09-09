WITH bounds AS (
  SELECT MIN(ts) AS trace_start FROM slice
), layers AS (
  SELECT
    id,
    name,
    ts,
    dur,
    ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) - 1 AS ordinal
  FROM slice
  WHERE name IN ('CheckpointFunction', 'CheckpointFunctionBackward')
)
SELECT
  name,
  ordinal,
  id,
  (ts - trace_start) / 1e6 AS start_ms,
  dur / 1e6 AS dur_ms
FROM layers, bounds
ORDER BY ts;
