SELECT id,ts/1e3 ts_us,dur/1e3 dur_us,name,EXTRACT_ARG(arg_set_id,'args.External id') external_id FROM slice WHERE category='cpu_op' AND dur>0;
