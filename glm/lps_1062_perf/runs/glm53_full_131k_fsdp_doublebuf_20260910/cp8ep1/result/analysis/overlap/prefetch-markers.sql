
SELECT name,MIN(ts) ts,MAX(ts+dur) en FROM slice
WHERE category='gpu_user_annotation' AND dur>0
AND (name GLOB 'block_*/forward' OR name GLOB 'block_*/backward_including_recompute'
 OR name GLOB 'layer_*/forward/*' OR name GLOB 'layer_*/recompute/*')
GROUP BY name ORDER BY ts
;
