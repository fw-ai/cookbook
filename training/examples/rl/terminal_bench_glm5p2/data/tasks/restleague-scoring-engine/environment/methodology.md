# REST League — Scoring Methodology (Internal Draft)

## Overview

The REST League ranks REST API black-box testing tools across three challenge
categories. Each tool runs against multiple target APIs for repeated sessions
within a fixed time budget.

## Data Sources

- **OpenAPI 3.0 specifications** define the available operations per API
  (located at `/app/data/specs/`)
- **HTTP session logs** are stored in the competition database at
  `/app/data/competition.db`
- **Configuration** parameters at `/app/data/config.yaml`
- Sessions may be repeated; some sessions may fail with `tool_crash` or
  `api_crash` status

## Metrics

### Operation Coverage

An "operation" is a unique `(HTTP method, path template)` pair defined in the
API's OpenAPI specification. A request is considered to have covered an operation
when:

1. The request's method and path match a defined operation in the spec
2. The response has a success status code (2XX range, i.e., 200-299)

Path parameters in templates (e.g., `{postId}`) match exactly one non-empty
path segment. When a request URL could match multiple path templates, the most
specific template takes precedence — specifically, the template with the greatest
number of literal (non-parameterized) path segments.

Only requests that match a defined operation contribute to any metric. Requests
to paths or methods not defined in the specification are silently discarded.

### Operation Coverage Rate

In addition to the raw count of covered operations, compute the **coverage rate**
as the proportion of operations covered relative to the API's total defined
operations. This normalizes across APIs of different sizes.

### Fault Detection

Each 5XX (500-599) response from a matched request with a non-empty error
message represents a potential fault. To avoid inflating counts from a single
underlying bug, error messages with sufficiently high normalized edit-distance-
based string similarity are consolidated as duplicates.

Errors are processed sequentially in timestamp order. Each new error message is
compared against all previously accepted distinct error messages within that
session. If the similarity to any existing error exceeds the deduplication
threshold, the new error is treated as a duplicate and not counted.

The specific similarity algorithm and threshold are not prescribed here; use the
calibration data at `/app/data/calibration.json` to determine appropriate
parameters.

### Efficiency (AUC)

For each session, track the cumulative count of newly covered operations (or
newly found unique faults) at each event timestamp. Compute the area under
this cumulative discovery curve from `t = 0` to the configured time budget.

The numerical integration method should be determined by calibrating against the
verified reference values.

## Aggregation

For each `(tool, API)` pair, average all metrics across valid repetitions.
Sessions with `tool_crash` or `api_crash` status are excluded from averaging.
If all repetitions for a pair are excluded, use 0 for all metrics.

Per-tool overall metrics should reflect each API's contribution proportional
to its complexity. The calibration data includes overall values to verify your
aggregation approach.

## Challenge Scoring

Standardize each aggregate metric to zero mean and unit variance across the
tool population. When the population variance is zero, all standardized values
are zero.

Challenge categories and badge assignments:

- **Bug Hunter** badge: tool with the highest standardized average fault count
- **Gold API Tester** badge: tool with the highest effectiveness score
  (a weighted combination of standardized fault count and standardized
  coverage rate, calibrated to prioritize fault detection)
- **Silver API Tester** badge: tool with the second highest effectiveness score
- **Roadrunner** badge: tool with the highest sum of standardized fault AUC +
  standardized operation AUC

The effectiveness score weights are not specified here. Use the calibration
data, which provides verified effectiveness scores for select tools, to
reverse-engineer the correct weighting.

## Calibration

A verified output subset with correct metric values is provided at
`/app/data/calibration.json`. Use it to validate your implementation and
determine any scoring parameters not fully specified in this document.
