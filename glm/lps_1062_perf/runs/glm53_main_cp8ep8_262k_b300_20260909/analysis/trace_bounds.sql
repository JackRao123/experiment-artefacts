SELECT
  COUNT(*) AS slices,
  MIN(ts) / 1e9 AS start_s,
  MAX(ts + dur) / 1e9 AS end_s
FROM slice;
