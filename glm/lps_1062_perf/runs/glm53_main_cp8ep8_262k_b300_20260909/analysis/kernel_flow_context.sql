SELECT
  k.id AS kernel_id,
  k.name AS kernel_name,
  launch.id AS launch_id,
  launch.name AS launch_name,
  launch.category AS launch_category,
  p1.name AS parent_1,
  p2.name AS parent_2,
  p3.name AS parent_3,
  p4.name AS parent_4,
  p5.name AS parent_5,
  p6.name AS parent_6
FROM flow f
JOIN slice launch ON launch.id = f.slice_out
JOIN slice k ON k.id = f.slice_in
LEFT JOIN slice p1 ON p1.id = launch.parent_id
LEFT JOIN slice p2 ON p2.id = p1.parent_id
LEFT JOIN slice p3 ON p3.id = p2.parent_id
LEFT JOIN slice p4 ON p4.id = p3.parent_id
LEFT JOIN slice p5 ON p5.id = p4.parent_id
LEFT JOIN slice p6 ON p6.id = p5.parent_id
WHERE k.category = 'kernel'
  AND (
    k.name GLOB '*hybrid_ep::*'
    OR k.name GLOB '*ncclDevKernel_ReduceScatter*'
    OR k.name GLOB '*ncclDevKernel_AllGather*'
    OR k.name GLOB '*ncclDevKernel_AllReduce*'
  )
LIMIT 100;
