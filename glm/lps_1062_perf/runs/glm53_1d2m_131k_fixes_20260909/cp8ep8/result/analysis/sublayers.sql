SELECT category,name,ts/1e3 ts_us,dur/1e6 ms FROM slice WHERE category='user_annotation' AND (name GLOB 'layer_*' OR name GLOB 'python_gc/*') ORDER BY ts;
