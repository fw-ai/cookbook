-- Diagnosis Co-occurrence Report
-- Find pairs of ICD codes that co-occur in the same patient.
-- Only include pairs appearing in at least 3 distinct patients.
-- List shared patient IDs as comma-separated string.
SELECT d1.icd_code AS code_a,
       d2.icd_code AS code_b,
       GROUP_CONCAT(DISTINCT d1.patient_id) AS shared_patients,
       COUNT(DISTINCT d1.patient_id) AS patient_count
FROM diagnoses d1
JOIN diagnoses d2 ON d1.patient_id = d2.patient_id
     AND d1.icd_code < d2.icd_code
GROUP BY d1.icd_code, d2.icd_code
HAVING patient_count >= 3
ORDER BY patient_count DESC;
