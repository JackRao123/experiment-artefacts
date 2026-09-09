WITH bounds AS (SELECT MIN(ts) AS trace_start FROM slice)
SELECT
  (k.ts - trace_start) / 1e6 AS start_ms,
  k.dur / 1e6 AS dur_ms,
  launch.name AS launch,
  p1.name AS parent_1,
  p2.name AS parent_2,
  p3.name AS parent_3,
  p4.name AS parent_4,
  p5.name AS parent_5,
  p6.name AS parent_6,
  p7.name AS parent_7,
  p8.name AS parent_8
FROM flow f
JOIN slice launch ON launch.id = f.slice_out
JOIN slice k ON k.id = f.slice_in
LEFT JOIN slice p1 ON p1.id = launch.parent_id
LEFT JOIN slice p2 ON p2.id = p1.parent_id
LEFT JOIN slice p3 ON p3.id = p2.parent_id
LEFT JOIN slice p4 ON p4.id = p3.parent_id
LEFT JOIN slice p5 ON p5.id = p4.parent_id
LEFT JOIN slice p6 ON p6.id = p5.parent_id
LEFT JOIN slice p7 ON p7.id = p6.parent_id
LEFT JOIN slice p8 ON p8.id = p7.parent_id
CROSS JOIN bounds
WHERE k.category = 'kernel'
  AND k.name GLOB '*AllReduce_Sum_f32*';
