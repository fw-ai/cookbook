-- Physician Revenue Summary
-- Total revenue and distinct procedure types per physician.
-- Include physicians with no procedures (show 0 revenue).
-- Handle NULL specialty by displaying 'Unassigned'.
SELECT ph.physician_id,
       ph.name || ' (' || IFNULL(ph.specialty, 'Unassigned') || ')' AS physician_label,
       d.name AS department,
       COUNT(pr.procedure_id) AS procedure_count,
       TOTAL(pr.cost) AS total_revenue,
       GROUP_CONCAT(DISTINCT pr.procedure_code) AS procedure_types
FROM physicians ph
JOIN departments d ON ph.department_id = d.department_id
LEFT JOIN procedures pr ON ph.physician_id = pr.physician_id
GROUP BY ph.physician_id, ph.name, ph.specialty, d.name
ORDER BY total_revenue DESC;
