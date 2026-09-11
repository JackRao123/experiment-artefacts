SELECT ts/1e3 ts_us,dur/1e3 dur_us,name FROM slice WHERE category='user_annotation' AND name GLOB 'block_*/*' AND dur>0;
