#!/usr/bin/env python3
"""Generate two source SQLite databases and reconciliation rules for the
cross-system clinical data warehouse reconciliation task."""
import sqlite3
import json
import os
from datetime import date, timedelta

# ---------- Overlap patients (N001-N010 <-> S001-S010) ----------
# (north_id, first, last, dob_iso, south_full_name, south_birth_mmddyyyy,
#  gender_n, sex_s, insurance_n, coverage_s)
OVERLAP = [
    ("N001", "James",    "Smith",    "1965-03-15", "James R Smith",    "03/15/1965", "M", "male",   "PPO",      "private_ppo"),
    ("N002", "Maria",    "Garcia",   "1972-07-22", "Maria L Garcia",   "07/22/1972", "F", "female", "HMO",      "private_hmo"),
    ("N003", "Robert",   "Johnson",  "1958-11-08", "Robert A Johnson", "11/08/1958", "M", "male",   "Medicare", "government_medicare"),
    ("N004", "Sarah",    "Williams", "1980-01-19", "Sarah M Williams", "01/19/1980", "F", "female", "Medicaid", "government_medicaid"),
    ("N005", "Michael",  "Brown",    "1975-06-30", "Michael T Brown",  "06/30/1975", "M", "male",   "Self-Pay", "self_pay"),
    ("N006", "Jennifer", "Jones",    "1968-09-14", "Jennifer K Jones", "09/14/1968", "F", "female", "PPO",      "private_ppo"),
    ("N007", "David",    "Davis",    "1982-04-05", "David P Davis",    "04/05/1982", "M", "male",   "HMO",      "private_hmo"),
    ("N008", "Lisa",     "Miller",   "1971-12-25", "Lisa N Miller",    "12/25/1971", "F", "female", "Medicare", "government_medicare"),
    ("N009", "Richard",  "Wilson",   "1963-08-17", "Richard E Wilson", "08/17/1963", "M", "male",   "PPO",      "private_ppo"),
    ("N010", "Patricia", "Moore",    "1977-02-28", "Patricia D Moore", "02/28/1977", "F", "female", "Medicaid", "government_medicaid"),
]

# ---------- North-only patients N011-N040 ----------
N_FIRST = [
    "Thomas", "Linda", "Christopher", "Barbara", "Daniel",
    "Susan", "Matthew", "Jessica", "Anthony", "Karen",
    "Mark", "Nancy", "Donald", "Betty", "Steven",
    "Margaret", "Paul", "Dorothy", "Andrew", "Sandra",
    "Joshua", "Ashley", "Kenneth", "Kimberly", "Kevin",
    "Emily", "Brian", "Donna", "George", "Michelle",
]
N_LAST = [
    "Taylor", "Anderson", "Thomas", "Jackson", "White",
    "Harris", "Martin", "Thompson", "Robinson", "Clark",
    "Lewis", "Lee", "Walker", "Hall", "Allen",
    "Young", "Hernandez", "King", "Wright", "Lopez",
    "Hill", "Scott", "Green", "Adams", "Baker",
    "Gonzalez", "Nelson", "Carter", "Mitchell", "Perez",
]

# ---------- South-only patients S011-S035 ----------
S_NAMES = [
    "Edward Turner", "Carol Campbell", "Frank Parker", "Helen Edwards",
    "Arthur Collins", "Ruth Stewart", "Roger Morris", "Ann Rogers",
    "Lawrence Reed", "Catherine Cook", "Wayne Morgan", "Jean Bell",
    "Jack Murphy", "Judith Bailey", "Terry Rivera",
    "Alice Cooper", "Raymond Hughes", "Marie Price", "Gerald Bennett",
    "Frances Wood", "Harold Barnes", "Ann Ross", "Roy Henderson",
    "Wanda Coleman", "Eugene Jenkins",
]

DEPARTMENTS = ["Emergency", "Cardiology", "Neurology", "Orthopedics", "Oncology"]

# ICD-10 codes (used by North)
ICD10 = [
    ("I10",     "Essential hypertension"),
    ("E11.9",   "Type 2 diabetes mellitus, unspecified"),
    ("J18.9",   "Pneumonia, unspecified organism"),
    ("K21.0",   "Gastro-esophageal reflux with esophagitis"),
    ("M54.5",   "Low back pain"),
    ("I21.9",   "Acute myocardial infarction, unspecified"),
    ("G43.909", "Migraine, unspecified, not intractable"),
    ("C34.90",  "Malignant neoplasm of bronchus/lung, unspecified"),
    ("J44.9",   "Chronic obstructive pulmonary disease, unspecified"),
    ("F32.9",   "Major depressive disorder, single episode"),
]

# ICD-9 codes (used by South) — map 1:1 to the ICD-10 list above
ICD9 = [
    ("401.9",  "Hypertension NOS"),
    ("250.00", "Diabetes mellitus type II"),
    ("486",    "Pneumonia organism NOS"),
    ("530.81", "Esophageal reflux"),
    ("724.5",  "Backache NOS"),
    ("410.9",  "AMI NOS"),
    ("346.90", "Migraine NOS"),
    ("162.9",  "Malign neoplasm bronchus/lung NOS"),
    ("496",    "Chronic airway obstruction NOS"),
    ("311",    "Depressive disorder NOS"),
]

# Lab definitions — North (SI/metric)
NLABS = [
    ("GLU",   "Glucose",             "mmol/L", 3.9,   5.6),
    ("CHOL",  "Total Cholesterol",   "mmol/L", 3.2,   5.2),
    ("HGB",   "Hemoglobin",          "g/L",    120.0, 175.0),
    ("CREAT", "Creatinine",          "umol/L", 44.0,  97.0),
    ("WBC",   "White Blood Cells",   "K/uL",   4.5,  11.0),
    ("PLT",   "Platelet Count",      "K/uL",  150.0, 400.0),
]

# Lab definitions — South (US customary for first 4, same units for last 2)
SLABS = [
    ("GLU",   "Glucose",             "mg/dL", 70.0,  100.0),
    ("CHOL",  "Cholesterol Total",   "mg/dL", 125.0, 200.0),
    ("HGB",   "Hemoglobin",          "g/dL",  12.0,  17.5),
    ("CREAT", "Creatinine",          "mg/dL", 0.5,   1.1),
    ("WBC",   "White Blood Cells",   "K/uL",  4.5,   11.0),
    ("PLT",   "Platelet Count",      "K/uL",  150.0, 400.0),
]

N_PHYSICIANS = [
    "Dr. Adams", "Dr. Bennett", "Dr. Carter", "Dr. Dawson", "Dr. Ellis",
    "Dr. Ford", "Dr. Grant", "Dr. Hayes", "Dr. Irving", "Dr. Jensen",
]
S_PROVIDERS = [
    "Dr. Kelly", "Dr. Lambert", "Dr. Morgan", "Dr. Norton", "Dr. Owens",
    "Dr. Palmer", "Dr. Quinn", "Dr. Roberts", "Dr. Stevens", "Dr. Tucker",
]


def _lab_value(ref_low, ref_high, idx, enc_idx):
    """Deterministic lab value generation."""
    mid = (ref_low + ref_high) / 2.0
    spread = ref_high - ref_low
    offset = ((idx * 17 + enc_idx * 13) % 100 - 50) / 50.0
    if idx % 3 == 0:
        val = ref_high + (idx % 15) * spread * 0.05 + spread * 0.1
    else:
        val = mid + offset * spread * 0.4
    val = round(val, 2)
    return max(val, 0.1)


# =====================================================================
#  NORTH DATABASE
# =====================================================================
def create_north(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("""CREATE TABLE patients (
        patient_id TEXT PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name  TEXT NOT NULL,
        dob        TEXT NOT NULL,
        gender     TEXT NOT NULL,
        insurance_type TEXT
    )""")
    c.execute("""CREATE TABLE encounters (
        encounter_id       TEXT PRIMARY KEY,
        patient_id         TEXT NOT NULL REFERENCES patients(patient_id),
        encounter_type     TEXT NOT NULL,
        department         TEXT NOT NULL,
        admit_date         TEXT NOT NULL,
        discharge_date     TEXT,
        attending_physician TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE diagnoses (
        diagnosis_id TEXT PRIMARY KEY,
        encounter_id TEXT NOT NULL REFERENCES encounters(encounter_id),
        icd10_code   TEXT NOT NULL,
        description  TEXT NOT NULL,
        severity     INTEGER NOT NULL CHECK(severity BETWEEN 1 AND 5),
        diagnosed_at TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE lab_results (
        lab_id       TEXT PRIMARY KEY,
        encounter_id TEXT NOT NULL REFERENCES encounters(encounter_id),
        test_code    TEXT NOT NULL,
        test_name    TEXT NOT NULL,
        value        REAL NOT NULL,
        unit         TEXT NOT NULL,
        ref_low      REAL NOT NULL,
        ref_high     REAL NOT NULL,
        collected_at TEXT NOT NULL
    )""")

    # --- Patients (40) ---
    patients = []
    ins_types = ["PPO", "HMO", "Medicare", "Medicaid", "Self-Pay"]
    for i, o in enumerate(OVERLAP):
        patients.append((o[0], o[1], o[2], o[3], o[6], o[8]))
    for i in range(30):
        pid = f"N{i+11:03d}"
        yr = 1950 + (i * 7) % 40
        mo = (i * 3) % 12 + 1
        dy = (i * 5) % 28 + 1
        patients.append((pid, N_FIRST[i], N_LAST[i],
                         f"{yr}-{mo:02d}-{dy:02d}",
                         "M" if i % 2 == 0 else "F",
                         ins_types[i % 5]))
    c.executemany("INSERT INTO patients VALUES(?,?,?,?,?,?)", patients)

    # --- Encounters (80 = 2 per patient) ---
    etypes = ["inpatient", "outpatient", "emergency"]
    encounters = []
    for pi in range(40):
        pid = patients[pi][0]
        for j in range(2):
            ei = pi * 2 + j
            eid = f"EN{ei+1:03d}"
            am = (ei * 3) % 12 + 1
            ad = (ei * 7) % 28 + 1
            admit = date(2023, am, ad)
            if ei % 5 == 0:
                dis = None
            else:
                dis = (admit + timedelta(days=(ei % 20) + 2)).isoformat()
            encounters.append((eid, pid, etypes[ei % 3], DEPARTMENTS[ei % 5],
                               admit.isoformat(), dis, N_PHYSICIANS[ei % 10]))
    c.executemany("INSERT INTO encounters VALUES(?,?,?,?,?,?,?)", encounters)

    # --- Diagnoses (160 = 2 per encounter) ---
    diagnoses = []
    for ei_idx in range(80):
        eid = encounters[ei_idx][0]
        for j in range(2):
            di = ei_idx * 2 + j
            did = f"ND{di+1:04d}"
            ix = (ei_idx + j) % 10
            diagnoses.append((did, eid, ICD10[ix][0], ICD10[ix][1],
                              (ei_idx + j) % 5 + 1, encounters[ei_idx][4]))
    c.executemany("INSERT INTO diagnoses VALUES(?,?,?,?,?,?)", diagnoses)

    # --- Lab results (240 = 3 per encounter) ---
    labs = []
    for ei_idx in range(80):
        eid = encounters[ei_idx][0]
        for j in range(3):
            li = ei_idx * 3 + j
            lid = f"NL{li+1:04d}"
            lx = (ei_idx + j * 2) % 6
            tc, tn, un, rl, rh = NLABS[lx]
            v = _lab_value(rl, rh, li, ei_idx)
            labs.append((lid, eid, tc, tn, v, un, rl, rh,
                         encounters[ei_idx][4]))
    c.executemany("INSERT INTO lab_results VALUES(?,?,?,?,?,?,?,?,?)", labs)

    conn.commit()
    conn.close()
    return len(patients), len(encounters), len(diagnoses), len(labs)


# =====================================================================
#  SOUTH DATABASE  (deliberately different schema)
# =====================================================================
def create_south(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("""CREATE TABLE registrations (
        mrn        TEXT PRIMARY KEY,
        full_name  TEXT NOT NULL,
        birth_date TEXT NOT NULL,
        sex        TEXT NOT NULL,
        coverage_plan TEXT
    )""")
    c.execute("""CREATE TABLE visits (
        visit_id        TEXT PRIMARY KEY,
        mrn             TEXT NOT NULL REFERENCES registrations(mrn),
        visit_type      TEXT NOT NULL,
        department_name TEXT NOT NULL,
        start_dt        TEXT NOT NULL,
        end_dt          TEXT,
        provider_name   TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE conditions (
        condition_id   TEXT PRIMARY KEY,
        visit_id       TEXT NOT NULL REFERENCES visits(visit_id),
        icd9_code      TEXT NOT NULL,
        condition_text TEXT NOT NULL,
        acuity         TEXT NOT NULL CHECK(acuity IN
                        ('low','moderate','high','severe','critical'))
    )""")
    c.execute("""CREATE TABLE laboratory (
        specimen_id   TEXT PRIMARY KEY,
        visit_id      TEXT NOT NULL REFERENCES visits(visit_id),
        analyte_code  TEXT NOT NULL,
        analyte_desc  TEXT NOT NULL,
        result_val    REAL NOT NULL,
        result_unit   TEXT NOT NULL,
        normal_low    REAL NOT NULL,
        normal_high   REAL NOT NULL,
        specimen_date TEXT NOT NULL
    )""")

    # --- Registrations (35) ---
    cov_plans = ["private_ppo", "private_hmo", "government_medicare",
                 "government_medicaid", "self_pay"]
    regs = []
    for i, o in enumerate(OVERLAP):
        regs.append((f"S{i+1:03d}", o[4], o[5], o[7], o[9]))
    for i in range(25):
        mrn = f"S{i+11:03d}"
        yr = 1945 + (i * 11) % 45
        mo = (i * 7) % 12 + 1
        dy = (i * 3) % 28 + 1
        regs.append((mrn, S_NAMES[i], f"{mo:02d}/{dy:02d}/{yr}",
                     "male" if i % 2 == 0 else "female",
                     cov_plans[i % 5]))
    c.executemany("INSERT INTO registrations VALUES(?,?,?,?,?)", regs)

    # --- Visits (70 = 2 per patient) ---
    vtypes = ["inpatient", "outpatient", "emergency"]
    visits = []
    for pi in range(35):
        mrn = regs[pi][0]
        for j in range(2):
            vi = pi * 2 + j
            vid = f"SV{vi+1:03d}"
            sm = (vi * 5) % 12 + 1
            sd = (vi * 11) % 28 + 1
            start = date(2023, sm, sd)
            start_s = f"{sm:02d}/{sd:02d}/2023"
            if vi % 6 == 0:
                end_s = None
            else:
                end = start + timedelta(days=(vi % 15) + 1)
                end_s = f"{end.month:02d}/{end.day:02d}/{end.year}"
            visits.append((vid, mrn, vtypes[vi % 3], DEPARTMENTS[vi % 5],
                           start_s, end_s, S_PROVIDERS[vi % 10]))
    c.executemany("INSERT INTO visits VALUES(?,?,?,?,?,?,?)", visits)

    # --- Conditions (140 = 2 per visit, ICD-9) ---
    acuities = ["low", "moderate", "high", "severe", "critical"]
    conditions = []
    for vi_idx in range(70):
        vid = visits[vi_idx][0]
        for j in range(2):
            ci = vi_idx * 2 + j
            cid = f"SC{ci+1:04d}"
            ix = (vi_idx + j) % 10
            conditions.append((cid, vid, ICD9[ix][0], ICD9[ix][1],
                               acuities[(vi_idx + j) % 5]))
    c.executemany("INSERT INTO conditions VALUES(?,?,?,?,?)", conditions)

    # --- Laboratory (210 = 3 per visit, US customary units) ---
    laboratory = []
    for vi_idx in range(70):
        vid = visits[vi_idx][0]
        for j in range(3):
            li = vi_idx * 3 + j
            lid = f"SL{li+1:04d}"
            lx = (vi_idx + j * 2) % 6
            tc, td, un, rl, rh = SLABS[lx]
            v = _lab_value(rl, rh, li, vi_idx)
            laboratory.append((lid, vid, tc, td, v, un, rl, rh,
                               visits[vi_idx][4]))
    c.executemany("INSERT INTO laboratory VALUES(?,?,?,?,?,?,?,?,?)",
                  laboratory)

    conn.commit()
    conn.close()
    return len(regs), len(visits), len(conditions), len(laboratory)


# =====================================================================
#  RECONCILIATION RULES
# =====================================================================
def create_rules(path):
    rules = {
        "patient_matching": {
            "description": "All three conditions must be satisfied simultaneously",
            "conditions": [
                {"field": "date_of_birth", "match_type": "exact"},
                {"field": "last_name", "match_type": "exact_case_insensitive"},
                {"field": "first_name", "match_type": "prefix_match",
                 "prefix_length": 3, "case_insensitive": True}
            ],
            "south_name_parsing": (
                "full_name format is 'FirstName [MiddleInitial] LastName' "
                "- first_name is the first space-delimited token, "
                "last_name is the last token"
            )
        },
        "icd9_to_icd10": {
            "401.9":  "I10",
            "250.00": "E11.9",
            "486":    "J18.9",
            "530.81": "K21.0",
            "724.5":  "M54.5",
            "410.9":  "I21.9",
            "346.90": "G43.909",
            "162.9":  "C34.90",
            "496":    "J44.9",
            "311":    "F32.9"
        },
        "unit_conversions": {
            "GLU":   {"from_unit": "mg/dL", "to_unit": "mmol/L",
                      "multiply_by": 0.0555},
            "CHOL":  {"from_unit": "mg/dL", "to_unit": "mmol/L",
                      "multiply_by": 0.0259},
            "HGB":   {"from_unit": "g/dL", "to_unit": "g/L",
                      "multiply_by": 10.0},
            "CREAT": {"from_unit": "mg/dL", "to_unit": "umol/L",
                      "multiply_by": 88.4}
        },
        "acuity_to_severity": {
            "low": 1, "moderate": 2, "high": 3, "severe": 4, "critical": 5
        },
        "gender_mapping": {"male": "M", "female": "F"},
        "insurance_mapping": {
            "private_ppo": "PPO",
            "private_hmo": "HMO",
            "government_medicare": "Medicare",
            "government_medicaid": "Medicaid",
            "self_pay": "Self-Pay"
        },
        "conflict_resolution": {
            "demographics": (
                "When a patient exists in both systems, use division_north "
                "values for first_name, last_name, gender, and insurance. "
                "Generate a new unified patient_id."
            ),
            "clinical_data": (
                "Union all encounters, diagnoses, and lab results from both "
                "systems without deduplication. Maintain source_division "
                "attribution on every clinical record."
            )
        }
    }
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)


def main():
    os.makedirs("/app", exist_ok=True)

    np, ne, nd, nl = create_north("/app/division_north.db")
    sp, sv, sc, slr = create_south("/app/division_south.db")
    create_rules("/app/reconciliation_rules.json")

    print(f"North: {np} patients, {ne} encounters, {nd} diagnoses, {nl} labs")
    print(f"South: {sp} patients, {sv} visits, {sc} conditions, {slr} labs")
    print(f"Expected unique patients after dedup: {np + sp - 10}")
    print(f"Expected total encounters: {ne + sv}")
    print(f"Expected total diagnoses: {nd + sc}")
    print(f"Expected total lab results: {nl + slr}")


if __name__ == "__main__":
    main()
