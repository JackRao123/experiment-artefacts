WITH target AS (
  SELECT
    k.id AS kernel_id,
    k.arg_set_id AS kernel_args,
    launch.parent_id AS annotation_id
  FROM flow f
  JOIN slice launch ON launch.id = f.slice_out
  JOIN slice k ON k.id = f.slice_in
  WHERE k.category = 'kernel' AND k.name GLOB '*AllReduce_Sum_f32*'
), ids AS (
  SELECT kernel_id AS slice_id, kernel_args AS arg_set_id, 'kernel' AS source FROM target
  UNION ALL
  SELECT s.id, s.arg_set_id, s.name
  FROM target t
  JOIN slice s ON s.id = t.annotation_id
  UNION ALL
  SELECT p.id, p.arg_set_id, p.name
  FROM target t
  JOIN slice s ON s.id = t.annotation_id
  JOIN slice p ON p.id = s.parent_id
)
SELECT
  ids.source,
  ids.slice_id,
  a.key,
  a.display_value
FROM ids
JOIN args a ON a.arg_set_id = ids.arg_set_id
ORDER BY ids.source, a.key;
