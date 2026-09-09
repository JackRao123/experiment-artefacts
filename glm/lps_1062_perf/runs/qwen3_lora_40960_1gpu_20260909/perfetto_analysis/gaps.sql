WITH k AS (SELECT ts,dur,name,MAX(ts+dur) OVER
        (ORDER BY ts,ts+dur ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_en
        FROM slice WHERE category='kernel' AND dur>0)
        SELECT ts, (ts-prev_en)/1e6 gap_ms,name next_kernel FROM k
        WHERE ts>prev_en ORDER BY gap_ms DESC LIMIT 30;
