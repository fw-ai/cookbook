-- seed data for utility_gis spatial database

CREATE TABLE zones (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    geom GEOMETRY(GEOMETRY)
);

INSERT INTO zones (name, geom) VALUES
('alpha',   ST_GeomFromText('POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))')),
('beta',    ST_GeomFromText('POLYGON((120 0, 220 0, 220 100, 120 100, 120 0))')),
('gamma',   ST_GeomFromText('POLYGON((90 40, 130 40, 130 60, 90 60, 90 40))')),
('delta',   ST_GeomFromText('POLYGON((300 0, 300 100, 400 0, 400 100, 300 0))')),
('epsilon', ST_GeomFromText('POLYGON((0 120, 50 120, 50 170, 0 170, 0 120))')),
('zeta',    ST_GeomFromText('POLYGON((30 150, 80 150, 80 200, 30 200, 30 150))')),
('eta',     ST_GeomFromText('POLYGON((20 20, 40 20, 40 40, 20 40, 20 20))'));

CREATE INDEX zones_geom_idx ON zones USING GIST (geom);

CREATE TABLE pipes (
    id INTEGER PRIMARY KEY,
    geom GEOMETRY(LINESTRING)
);

INSERT INTO pipes (id, geom) VALUES
(1,  ST_GeomFromText('LINESTRING(10 50, 30 50)')),
(2,  ST_GeomFromText('LINESTRING(30 50, 50 50)')),
(3,  ST_GeomFromText('LINESTRING(50 50, 70 50)')),
(4,  ST_GeomFromText('LINESTRING(70 50, 90 50)')),
(5,  ST_GeomFromText('LINESTRING(90 50, 110 50)')),
(6,  ST_GeomFromText('LINESTRING(30 50, 30 70)')),
(7,  ST_GeomFromText('LINESTRING(30 70, 30 90)')),
(8,  ST_GeomFromText('LINESTRING(30 70, 50 70)')),
(9,  ST_GeomFromText('LINESTRING(70 50, 70 30)')),
(10, ST_GeomFromText('LINESTRING(70 30, 90 30)')),
(11, ST_GeomFromText('LINESTRING(90 50, 90 70)')),
(12, ST_GeomFromText('LINESTRING(200 200, 220 200)')),
(13, ST_GeomFromText('LINESTRING(220 200, 240 200)')),
(14, ST_GeomFromText('LINESTRING(220 200, 220 220)')),
(15, ST_GeomFromText('LINESTRING(40 40, 40 60)'));

CREATE INDEX pipes_geom_idx ON pipes USING GIST (geom);

CREATE TABLE wells (
    id INTEGER PRIMARY KEY,
    geom GEOMETRY(POINT)
);

INSERT INTO wells (id, geom) VALUES
(1, ST_GeomFromText('POINT(15 53)')),
(2, ST_GeomFromText('POINT(45 65)')),
(3, ST_GeomFromText('POINT(75 50)')),
(4, ST_GeomFromText('POINT(215 203)'));

CREATE INDEX wells_geom_idx ON wells USING GIST (geom);
