-- Abnormal Lab Result Trends
-- Monthly abnormal lab result rates by test type.
-- A result is abnormal if its value falls outside [reference_low, reference_high].
-- Only include test/month combinations where the abnormal rate exceeds 15%.
SELECT SUBSTR(lr.result_date, 1, 7) AS month,
       lr.test_code,
       lr.test_name,
       COUNT(*) AS abnormal_count,
       ROUND(100.0 * COUNT(*) / t.total, 1) AS abnormal_pct
FROM lab_results lr
JOIN (
    SELECT test_code, SUBSTR(result_date, 1, 7) AS month, COUNT(*) AS total
    FROM lab_results
    GROUP BY test_code, SUBSTR(result_date, 1, 7)
) t ON lr.test_code = t.test_code AND SUBSTR(lr.result_date, 1, 7) = t.month
WHERE CAST(lr.value AS REAL) NOT BETWEEN lr.reference_low AND lr.reference_high
GROUP BY SUBSTR(lr.result_date, 1, 7), lr.test_code, lr.test_name, t.total
HAVING abnormal_pct > 15.0
ORDER BY month, abnormal_pct DESC;
