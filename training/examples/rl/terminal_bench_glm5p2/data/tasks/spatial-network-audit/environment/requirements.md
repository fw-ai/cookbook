# Utility Network Annual Report — Metric Specifications

## 1. Zone Boundary Conflicts (`zone_conflict_count`)
Count the number of distinct pairs of service zones whose territories share two-dimensional area, indicating boundary survey discrepancies requiring field verification.

## 2. Total Service Area (`total_zone_area`)
Compute the aggregate geometric area of all service zones. All zone geometries must represent valid spatial regions for accurate area measurement.

## 3. Directed Network Reachability (`network_reach_count`, `network_reach_length`)
Starting from pipe id=1, walk the pipe network by following connections at shared junction points between pipe segments. Report both the count of distinct reachable pipes and their total combined length.

## 4. Facility Assignments (`well_assignments`)
Assign each well to its nearest pipe based on the geometric distance from the well to the pipe segment. For each assignment, compute the linear referencing fraction indicating where along the pipe the well projects. Output as an array of `[well_id, pipe_id, fraction]` triples.

## 5. Pipe Coverage in Zone Alpha (`alpha_pipe_coverage`)
Compute the total length of pipe infrastructure serving the 'alpha' service zone. Include proportional lengths for pipes that partially cross the zone boundary.
