-- JCGM 101:2008 Measurement Uncertainty Problems
-- Input data for gauge block calibration and comparison loss

CREATE TABLE models (
    problem_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    model_equation TEXT NOT NULL,
    notes TEXT
);

INSERT INTO models VALUES (
    'gauge_block',
    'Gauge block calibration (JCGM 101:2008 Section 9.5)',
    'delta_L = L_s + D + d1 + d2 - L_s * (delta_alpha * (theta_0 + Delta) + alpha_s * delta_theta) - L_nom',
    'L_nom is the nominal length of the gauge block in nanometers. Nine input quantities from diverse distributions.'
);

INSERT INTO models VALUES (
    'comparison_loss',
    'Comparison loss in microwave power meter calibration (JCGM 101:2008 Section 9.4)',
    'delta_Y = X1^2 + X2^2',
    '(X1, X2) follow a bivariate Gaussian distribution parameterized in comparison_loss_cases. Some configurations are pathological for first-order linearization.'
);

CREATE TABLE gauge_block_config (
    L_nom_nm REAL NOT NULL,
    coverage_probability REAL NOT NULL,
    n_sig_digits INTEGER NOT NULL,
    validation_n_sig_digits INTEGER NOT NULL
);

INSERT INTO gauge_block_config VALUES (50000000, 0.99, 2, 1);

CREATE TABLE gauge_block_inputs (
    name TEXT PRIMARY KEY,
    mcm_distribution TEXT NOT NULL,
    mcm_params_json TEXT NOT NULL,
    guf_estimate REAL NOT NULL,
    guf_std_uncertainty REAL NOT NULL,
    guf_dof INTEGER,
    unit TEXT NOT NULL
);

INSERT INTO gauge_block_inputs VALUES ('L_s', 'scaled_shifted_t', '{"mu": 50000623, "sigma": 25, "nu": 18}', 50000623, 25, 18, 'nm');
INSERT INTO gauge_block_inputs VALUES ('D', 'scaled_shifted_t', '{"mu": 215, "sigma": 6, "nu": 24}', 215, 6, 24, 'nm');
INSERT INTO gauge_block_inputs VALUES ('d1', 'scaled_shifted_t', '{"mu": 0, "sigma": 4, "nu": 5}', 0, 4, 5, 'nm');
INSERT INTO gauge_block_inputs VALUES ('d2', 'scaled_shifted_t', '{"mu": 0, "sigma": 7, "nu": 8}', 0, 7, 8, 'nm');
INSERT INTO gauge_block_inputs VALUES ('alpha_s', 'rectangular', '{"a": 9.5e-6, "b": 13.5e-6}', 11.5e-6, 1.1547005383792517e-6, NULL, 'per_degC');
INSERT INTO gauge_block_inputs VALUES ('theta_0', 'gaussian', '{"mu": -0.1, "sigma": 0.2}', -0.1, 0.2, NULL, 'degC');
INSERT INTO gauge_block_inputs VALUES ('Delta', 'arcsine', '{"a": -0.5, "b": 0.5}', 0.0, 0.35355339059327373, NULL, 'degC');
INSERT INTO gauge_block_inputs VALUES ('delta_alpha', 'curvilinear_trapezoidal', '{"a": -1.0e-6, "b": 1.0e-6, "d": 0.1e-6}', 0.0, 5.773502691896258e-7, 50, 'per_degC');
INSERT INTO gauge_block_inputs VALUES ('delta_theta', 'curvilinear_trapezoidal', '{"a": -0.050, "b": 0.050, "d": 0.025}', 0.0, 0.028867513459481287, 2, 'degC');

CREATE TABLE comparison_loss_config (
    coverage_probability REAL NOT NULL,
    n_sig_digits INTEGER NOT NULL,
    validation_n_sig_digits INTEGER NOT NULL
);

INSERT INTO comparison_loss_config VALUES (0.95, 1, 1);

CREATE TABLE comparison_loss_cases (
    case_id INTEGER PRIMARY KEY,
    x1 REAL NOT NULL,
    x2 REAL NOT NULL,
    u_x1 REAL NOT NULL,
    u_x2 REAL NOT NULL,
    r REAL NOT NULL
);

INSERT INTO comparison_loss_cases VALUES (1, 0.0, 0.0, 0.005, 0.005, 0.0);
INSERT INTO comparison_loss_cases VALUES (2, 0.010, 0.0, 0.005, 0.005, 0.0);
INSERT INTO comparison_loss_cases VALUES (3, 0.050, 0.0, 0.005, 0.005, 0.0);
INSERT INTO comparison_loss_cases VALUES (4, 0.0, 0.0, 0.005, 0.005, 0.9);
INSERT INTO comparison_loss_cases VALUES (5, 0.010, 0.0, 0.005, 0.005, 0.9);
INSERT INTO comparison_loss_cases VALUES (6, 0.050, 0.0, 0.005, 0.005, 0.9);
