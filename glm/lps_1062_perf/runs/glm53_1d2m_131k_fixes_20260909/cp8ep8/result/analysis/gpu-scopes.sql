WITH regions AS MATERIALIZED (
 SELECT name,track_id,ts,ts+dur en FROM slice
 WHERE category='gpu_user_annotation' AND dur>0
 AND (name GLOB 'layer_*/forward/*' OR name GLOB 'layer_*/recompute/*'))
SELECT r.name scope,k.id,k.ts/1e3 ts_us,k.dur/1e3 dur_us,k.name,k.track_id
FROM regions r JOIN slice k ON k.track_id=r.track_id AND k.ts>=r.ts AND k.ts+k.dur<=r.en
WHERE k.category='kernel' AND k.dur>0 ORDER BY r.name,k.ts;
