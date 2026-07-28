-- Monthly Revenue Pivot by Department
-- Conditional aggregation to show monthly revenue broken down by department.
-- Columns for Cardiology, Neurology, Oncology, and Surgery departments.
-- Total revenue column includes all departments.
SELECT strftime('%Y', pr.procedure_date) AS year,
       strftime('%m', pr.procedure_date) AS month,
       SUM(IIF(d.name = 'Cardiology', pr.cost, 0)) AS cardiology_rev,
       SUM(IIF(d.name = 'Neurology', pr.cost, 0)) AS neurology_rev,
       SUM(IIF(d.name = 'Oncology', pr.cost, 0)) AS oncology_rev,
       SUM(IIF(d.name = 'Surgery', pr.cost, 0)) AS surgery_rev,
       SUM(pr.cost) AS total_rev
FROM procedures pr
JOIN physicians ph ON pr.physician_id = ph.physician_id
JOIN departments d ON ph.department_id = d.department_id
GROUP BY strftime('%Y', pr.procedure_date), strftime('%m', pr.procedure_date)
ORDER BY year, month;
