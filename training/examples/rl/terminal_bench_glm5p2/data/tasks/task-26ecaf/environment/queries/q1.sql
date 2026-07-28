-- Department Risk Stratification
-- Compute risk level per department based on average diagnosis severity.
-- High Risk if average severity > 3.5. Only include departments with >= 5 distinct patients.
-- Include total procedure cost per department (0 if no procedures).
SELECT d.name AS department,
       IIF(AVG(diag.severity) > 3.5, 'High Risk', 'Standard') AS risk_level,
       COUNT(DISTINCT p.patient_id) AS patient_count,
       ROUND(AVG(diag.severity), 2) AS avg_severity,
       IFNULL(SUM(pr.cost), 0) AS total_procedure_cost
FROM departments d
JOIN patients p ON d.department_id = p.department_id
JOIN diagnoses diag ON p.patient_id = diag.patient_id
LEFT JOIN procedures pr ON diag.diagnosis_id = pr.diagnosis_id
GROUP BY d.department_id, d.name
HAVING patient_count >= 5
ORDER BY avg_severity DESC;
