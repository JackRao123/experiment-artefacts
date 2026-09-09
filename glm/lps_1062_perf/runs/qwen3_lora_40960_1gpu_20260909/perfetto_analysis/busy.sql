WITH k AS (SELECT ts,ts+dur en,dur FROM slice
          WHERE category='kernel' AND dur>0),
        ordered AS (SELECT *,MAX(en) OVER (ORDER BY ts,en ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_en FROM k)
        SELECT COUNT(*) kernels,(MAX(en)-MIN(ts))/1e6 span_ms,SUM(dur)/1e6 sum_ms,
        SUM(CASE WHEN prev_en IS NULL THEN en-ts WHEN en>prev_en THEN en-MAX(ts,prev_en) ELSE 0 END)/1e6 union_busy_ms,
        SUM(CASE WHEN ts>prev_en THEN ts-prev_en ELSE 0 END)/1e6 gaps_ms,
        MAX(CASE WHEN ts>prev_en THEN ts-prev_en ELSE 0 END)/1e6 max_gap_ms FROM ordered;
