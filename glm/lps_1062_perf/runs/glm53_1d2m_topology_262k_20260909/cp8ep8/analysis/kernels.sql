SELECT id,ts/1e3 ts_us,dur/1e3 dur_us,name,track_id,EXTRACT_ARG(arg_set_id,'args.External id') external_id FROM slice WHERE category='kernel' AND dur>0 ORDER BY ts;
