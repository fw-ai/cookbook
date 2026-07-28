-- REST League 2026 Competition Database
-- Contains session recordings from the API testing tool competition

CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    tool TEXT NOT NULL,
    api TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed'
);

CREATE TABLE interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    error_message TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE tool_info (
    tool TEXT PRIMARY KEY,
    version TEXT,
    language TEXT,
    description TEXT
);

CREATE TABLE competition_runs (
    id INTEGER PRIMARY KEY,
    run_date TEXT NOT NULL,
    round TEXT NOT NULL,
    description TEXT
);

CREATE TABLE api_metadata (
    api TEXT PRIMARY KEY,
    framework TEXT,
    loc INTEGER,
    database_backend TEXT
);

-- Tool registry
INSERT INTO tool_info VALUES ('AlphaTester', '2.1.0', 'Java', 'Property-based REST API testing');
INSERT INTO tool_info VALUES ('BetaFuzz', '0.9.3', 'Python', 'Mutation-guided API fuzzer');
INSERT INTO tool_info VALUES ('GammaProbe', '1.5.2', 'Go', 'Grammar-based API exploration');
INSERT INTO tool_info VALUES ('DeltaScan', '3.0.1', 'Rust', 'Coverage-guided endpoint scanner');
INSERT INTO tool_info VALUES ('EpsilonBot', '0.2.0', 'TypeScript', 'LLM-guided API tester (withdrew before finals)');

-- Competition schedule
INSERT INTO competition_runs VALUES (1, '2026-03-15', 'qualifying', 'First qualifying round');
INSERT INTO competition_runs VALUES (2, '2026-03-22', 'qualifying', 'Second qualifying round');
INSERT INTO competition_runs VALUES (3, '2026-04-05', 'finals', 'Final round at ICSE 2026');

-- API metadata
INSERT INTO api_metadata VALUES ('blog', 'Spring Boot', 12450, 'PostgreSQL');
INSERT INTO api_metadata VALUES ('shop', 'Spring Boot', 18200, 'MySQL');
INSERT INTO api_metadata VALUES ('hospital', 'Quarkus', 24100, 'PostgreSQL');

-- ================================================================
-- Session data
-- ================================================================

-- AlphaTester sessions
INSERT INTO sessions VALUES (1, 'AlphaTester', 'blog', 1, 'completed');
INSERT INTO sessions VALUES (2, 'AlphaTester', 'blog', 2, 'completed');
INSERT INTO sessions VALUES (3, 'AlphaTester', 'shop', 1, 'completed');
INSERT INTO sessions VALUES (4, 'AlphaTester', 'shop', 2, 'completed');

-- BetaFuzz sessions
INSERT INTO sessions VALUES (5, 'BetaFuzz', 'blog', 1, 'completed');
INSERT INTO sessions VALUES (6, 'BetaFuzz', 'blog', 2, 'completed');
INSERT INTO sessions VALUES (7, 'BetaFuzz', 'shop', 1, 'completed');
INSERT INTO sessions VALUES (8, 'BetaFuzz', 'shop', 2, 'completed');

-- GammaProbe sessions
INSERT INTO sessions VALUES (9, 'GammaProbe', 'blog', 1, 'tool_crash');
INSERT INTO sessions VALUES (10, 'GammaProbe', 'blog', 2, 'completed');
INSERT INTO sessions VALUES (11, 'GammaProbe', 'shop', 1, 'completed');
INSERT INTO sessions VALUES (12, 'GammaProbe', 'shop', 2, 'completed');

-- DeltaScan sessions
INSERT INTO sessions VALUES (13, 'DeltaScan', 'blog', 1, 'completed');
INSERT INTO sessions VALUES (14, 'DeltaScan', 'blog', 2, 'completed');
INSERT INTO sessions VALUES (15, 'DeltaScan', 'shop', 1, 'api_crash');
INSERT INTO sessions VALUES (16, 'DeltaScan', 'shop', 2, 'completed');

-- Hospital sessions
INSERT INTO sessions VALUES (17, 'AlphaTester', 'hospital', 1, 'completed');
INSERT INTO sessions VALUES (18, 'AlphaTester', 'hospital', 2, 'completed');
INSERT INTO sessions VALUES (19, 'BetaFuzz', 'hospital', 1, 'completed');
INSERT INTO sessions VALUES (20, 'BetaFuzz', 'hospital', 2, 'completed');
INSERT INTO sessions VALUES (21, 'GammaProbe', 'hospital', 1, 'completed');
INSERT INTO sessions VALUES (22, 'GammaProbe', 'hospital', 2, 'completed');
INSERT INTO sessions VALUES (23, 'DeltaScan', 'hospital', 1, 'completed');
INSERT INTO sessions VALUES (24, 'DeltaScan', 'hospital', 2, 'completed');

-- ================================================================
-- Interaction data
-- ================================================================

-- Session 1: AlphaTester / blog / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 10, 'POST', '/auth/login', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 30, 'GET', '/posts', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 50, 'POST', '/posts', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 70, 'GET', '/posts/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 100, 'PUT', '/posts/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 130, 'GET', '/posts/1/comments', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 200, 'POST', '/posts', 500, 'NullPointerException in PostService.create at line 42');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 250, 'GET', '/posts/999', 500, 'NullPointerException in PostService.create at line 43');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 270, 'PUT', '/posts/1', 500, 'ValidationException: field ''title'' is required');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (1, 280, 'POST', '/admin/hack', 500, 'SecurityException: unauthorized access attempt');

-- Session 2: AlphaTester / blog / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (2, 15, 'POST', '/auth/login', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (2, 35, 'GET', '/posts', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (2, 55, 'POST', '/posts', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (2, 80, 'GET', '/posts/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (2, 110, 'PUT', '/posts/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (2, 140, 'GET', '/posts/2/comments', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (2, 210, 'POST', '/posts', 500, 'NullPointerException in PostService.create at line 42');

-- Session 3: AlphaTester / shop / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (3, 10, 'POST', '/auth/register', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (3, 30, 'GET', '/products', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (3, 50, 'POST', '/products', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (3, 80, 'GET', '/products/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (3, 110, 'DELETE', '/products/1', 204, NULL);

-- Session 4: AlphaTester / shop / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (4, 15, 'POST', '/auth/register', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (4, 40, 'GET', '/products', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (4, 60, 'POST', '/products', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (4, 90, 'GET', '/products/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (4, 120, 'DELETE', '/products/2', 204, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (4, 200, 'POST', '/products', 500, 'ConstraintViolationException: unique key violation on email');

-- Session 5: BetaFuzz / blog / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 5, 'GET', '/posts', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 8, 'GET', '/posts/featured', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 10, 'POST', '/posts', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 15, 'GET', '/posts/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 25, 'POST', '/posts', 500, 'NullPointerException in PostService.create at line 42');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 35, 'GET', '/posts/1', 500, 'NullPointerException in PostService.create at line 42');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 45, 'PUT', '/posts/1', 500, 'ValidationException: field ''title'' is required');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 55, 'POST', '/auth/login', 500, 'ConnectionTimeoutException: database pool exhausted');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (5, 65, 'GET', '/posts/2/comments', 500, 'IllegalArgumentException: invalid ID format');

-- Session 6: BetaFuzz / blog / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (6, 5, 'GET', '/posts', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (6, 8, 'GET', '/posts/featured', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (6, 10, 'POST', '/posts', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (6, 20, 'GET', '/posts/3', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (6, 30, 'POST', '/posts', 500, 'NullPointerException in PostService.create at line 42');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (6, 40, 'PUT', '/posts/3', 500, 'ValidationException: field ''title'' is required');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (6, 50, 'POST', '/auth/login', 500, 'ConnectionTimeoutException: database pool exhausted');

-- Session 7: BetaFuzz / shop / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (7, 10, 'GET', '/products', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (7, 20, 'POST', '/products', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (7, 30, 'GET', '/products/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (7, 40, 'POST', '/products', 500, 'ConstraintViolationException: unique key violation on email');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (7, 50, 'DELETE', '/products/99', 500, 'ResourceNotFoundException: product with id 99 not found');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (7, 55, 'DELETE', '/products/100', 500, 'ResourceNotFoundException: product with id 100 not found');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (7, 60, 'POST', '/auth/register', 500, 'IllegalStateException: registration service unavailable');

-- Session 8: BetaFuzz / shop / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (8, 10, 'GET', '/products', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (8, 20, 'POST', '/products', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (8, 35, 'GET', '/products/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (8, 45, 'POST', '/products', 500, 'ConstraintViolationException: unique key violation on email');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (8, 55, 'DELETE', '/products/100', 500, 'ResourceNotFoundException: product with id 100 not found');

-- Session 9: GammaProbe / blog / rep 1 (tool_crash)
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (9, 5, 'GET', '/posts', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (9, 10, 'POST', '/posts', 201, NULL);

-- Session 10: GammaProbe / blog / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (10, 10, 'POST', '/auth/login', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (10, 30, 'GET', '/posts', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (10, 50, 'POST', '/posts', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (10, 80, 'GET', '/posts/4', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (10, 150, 'POST', '/posts', 500, 'NullPointerException in PostService.create at line 42');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (10, 160, 'DELETE', '/posts', 500, 'MethodNotAllowedException: DELETE not supported on /posts');

-- Session 11: GammaProbe / shop / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (11, 10, 'POST', '/auth/register', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (11, 30, 'GET', '/products', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (11, 50, 'POST', '/products', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (11, 70, 'GET', '/products/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (11, 100, 'DELETE', '/products/1', 500, 'ResourceNotFoundException: product with id 99 not found');

-- Session 12: GammaProbe / shop / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (12, 15, 'POST', '/auth/register', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (12, 40, 'GET', '/products', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (12, 60, 'POST', '/products', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (12, 85, 'GET', '/products/3', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (12, 110, 'DELETE', '/products/3', 204, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (12, 180, 'POST', '/products', 500, 'ConstraintViolationException: unique key violation on email');

-- Session 13: DeltaScan / blog / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (13, 5, 'GET', '/posts', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (13, 15, 'POST', '/posts', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (13, 30, 'GET', '/posts/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (13, 50, 'GET', '/posts/featured', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (13, 100, 'PUT', '/posts/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (13, 200, 'POST', '/posts', 500, 'NullPointerException in PostService.create at line 42');

-- Session 14: DeltaScan / blog / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (14, 10, 'GET', '/posts', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (14, 25, 'POST', '/posts', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (14, 40, 'GET', '/posts/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (14, 70, 'PUT', '/posts/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (14, 150, 'POST', '/posts', 500, 'NullPointerException in PostService.create at line 42');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (14, 200, 'GET', '/posts/2/comments', 200, NULL);

-- Session 15: DeltaScan / shop / rep 1 (api_crash)
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (15, 10, 'GET', '/products', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (15, 20, 'POST', '/products', 201, NULL);

-- Session 16: DeltaScan / shop / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (16, 10, 'POST', '/auth/register', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (16, 30, 'GET', '/products', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (16, 50, 'POST', '/products', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (16, 70, 'GET', '/products/1', 200, NULL);

-- Session 17: AlphaTester / hospital / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 10, 'POST', '/auth/token', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 30, 'GET', '/patients', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 50, 'POST', '/patients', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 80, 'GET', '/patients/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 100, 'PUT', '/patients/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 130, 'GET', '/patients/1/records', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 160, 'POST', '/patients/1/records', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 200, 'GET', '/patients/1/records/summary', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 230, 'GET', '/patients/1/records/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 260, 'GET', '/patients/1/records/1/attachments', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (17, 280, 'POST', '/patients/1/records', 500, 'DataIntegrityException: duplicate record identifier r-001');

-- Session 18: AlphaTester / hospital / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 15, 'POST', '/auth/token', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 35, 'GET', '/patients', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 55, 'POST', '/patients', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 85, 'GET', '/patients/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 115, 'PUT', '/patients/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 145, 'GET', '/patients/2/records', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 175, 'POST', '/patients/2/records', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 195, 'GET', '/patients/2/records/summary', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 220, 'GET', '/patients/2/records/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (18, 250, 'DELETE', '/patients/2/records/2', 500, 'ConcurrencyException: record locked by another transaction');

-- Session 19: BetaFuzz / hospital / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 5, 'GET', '/patients', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 10, 'POST', '/patients', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 20, 'GET', '/patients/search', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 30, 'GET', '/patients/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 40, 'GET', '/patients/1/records', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 50, 'POST', '/patients/1/records', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 55, 'GET', '/patients/1/records/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 60, 'POST', '/patients/1/records', 500, 'DataIntegrityException: duplicate record identifier r-002');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 70, 'GET', '/patients/1/records/summary', 500, 'NullPointerException: patient record aggregation failed');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 80, 'DELETE', '/patients/1/records/1', 500, 'ConcurrencyException: record locked by another transaction');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (19, 90, 'POST', '/auth/token', 500, 'AuthenticationException: invalid credentials format');

-- Session 20: BetaFuzz / hospital / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 5, 'GET', '/patients', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 15, 'POST', '/patients', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 25, 'GET', '/patients/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 35, 'GET', '/patients/2/records', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 45, 'POST', '/patients/2/records', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 55, 'GET', '/patients/2/records/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 65, 'POST', '/patients/2/records', 500, 'DataIntegrityException: duplicate record identifier r-003');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 75, 'DELETE', '/patients/2/records/2', 500, 'ConcurrencyException: record locked by another transaction');
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (20, 85, 'GET', '/patients/search', 500, 'InvalidQueryException: search parameter ''q'' required');

-- Session 21: GammaProbe / hospital / rep 1
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (21, 10, 'POST', '/auth/token', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (21, 30, 'GET', '/patients', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (21, 50, 'POST', '/patients', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (21, 70, 'GET', '/patients/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (21, 100, 'GET', '/patients/1/records', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (21, 130, 'GET', '/patients/1/records/summary', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (21, 180, 'POST', '/patients/1/records', 500, 'DataIntegrityException: duplicate record identifier r-004');

-- Session 22: GammaProbe / hospital / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 15, 'POST', '/auth/token', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 40, 'GET', '/patients', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 60, 'POST', '/patients', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 80, 'GET', '/patients/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 110, 'GET', '/patients/2/records', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 140, 'POST', '/patients/2/records', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 170, 'GET', '/patients/2/records/summary', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 200, 'GET', '/patients/2/records/2', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 230, 'GET', '/patients/2/records/2/attachments', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (22, 260, 'DELETE', '/patients/2/records/2', 500, 'ConcurrencyException: record locked by another transaction');

-- Session 23: DeltaScan / hospital / rep 1 (completed, 0 interactions)
-- No interactions recorded (tool started but performed no requests)

-- Session 24: DeltaScan / hospital / rep 2
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (24, 10, 'POST', '/auth/token', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (24, 30, 'GET', '/patients', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (24, 50, 'POST', '/patients', 201, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (24, 70, 'GET', '/patients/1', 200, NULL);
INSERT INTO interactions (session_id, timestamp, method, path, status_code, error_message) VALUES (24, 100, 'GET', '/patients/1/records', 200, NULL);

-- Indexes for query performance
CREATE INDEX idx_interactions_session ON interactions(session_id);
CREATE INDEX idx_sessions_tool_api ON sessions(tool, api);
CREATE INDEX idx_interactions_timestamp ON interactions(session_id, timestamp);
