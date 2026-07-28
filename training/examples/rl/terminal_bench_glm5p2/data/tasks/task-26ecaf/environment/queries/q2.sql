-- Length of Stay Analysis
-- Average and maximum length of stay (in days) per department.
-- Only include patients who have been discharged (non-NULL discharge_date).
-- Results ordered by average stay descending.
SELECT p.department_id,
       d.name AS department,
       CAST(AVG(julianday(p.discharge_date) - julianday(p.admission_date)) AS INTEGER) AS avg_stay_days,
       MAX(CAST(julianday(p.discharge_date) - julianday(p.admission_date) AS INTEGER)) AS max_stay,
       COUNT(*) AS discharged_count
FROM patients p
JOIN departments d ON p.department_id = d.department_id
WHERE p.discharge_date IS NOT NULL
GROUP BY p.department_id, d.name
ORDER BY avg_stay_days DESC;
