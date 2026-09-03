SELECT
  (SELECT COUNT(*) FROM slice WHERE name = 'CheckpointFunction') AS layers,
  (SELECT SUM(dur) FROM slice WHERE name = 'CheckpointFunction') /
    (SELECT COUNT(*) FROM slice WHERE name = 'CheckpointFunction') / 1e6 AS fwd_per_layer_ms,
  (SELECT SUM(dur) FROM slice WHERE name = 'CheckpointFunctionBackward') /
    (SELECT COUNT(*) FROM slice WHERE name = 'CheckpointFunctionBackward') / 1e6 AS bwd_per_layer_ms;
