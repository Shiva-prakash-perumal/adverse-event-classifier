"""
ingestion.py
------------
Downloads and loads FDA FAERS (Adverse Event Reporting System) data.
FAERS is a public database maintained by the FDA containing adverse event
reports submitted by healthcare professionals and consumers.

Data source: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
"""

import glob
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# =============================================================================
# SYNTHETIC DATA GENERATION — COMMENTED OUT (using real FAERS data now)
# Uncomment and call generate_synthetic_faers() if you need offline testing
# =============================================================================

# def generate_synthetic_faers(n_samples: int = 5000, save: bool = True) -> pd.DataFrame:
#     np.random.seed(42)
#     logger.info(f"Generating {n_samples} synthetic FAERS records...")
#     ages = np.random.normal(55, 18, n_samples).clip(18, 90).astype(int)
#     genders = np.random.choice(["Male", "Female", "Unknown"], n_samples, p=[0.45, 0.45, 0.10])
#     weights = np.random.normal(75, 18, n_samples).clip(40, 150).round(1)
#     drug_names = np.random.choice(["DrugA", "DrugB", "DrugC", "DrugD", "DrugE"], n_samples)
#     dosages = np.random.choice([10, 25, 50, 100, 200, 500], n_samples)
#     routes = np.random.choice(["Oral", "Intravenous", "Subcutaneous", "Topical"], n_samples, p=[0.60, 0.25, 0.10, 0.05])
#     adverse_events = np.random.choice(["Nausea", "Headache", "Chest Pain", "Dyspnea", "Rash", "Fatigue", "Dizziness", "Vomiting", "Anaphylaxis", "Cardiac Arrest", "Liver Failure", "Hypertension", "Hypotension", "Fever"], n_samples)
#     time_to_onset = np.random.exponential(scale=5, size=n_samples).clip(0.1, 90).round(1)
#     num_concomitant_drugs = np.random.poisson(lam=2, size=n_samples).clip(0, 10)
#     symptom_count = np.random.randint(1, 6, n_samples)
#     has_comorbidity = np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
#     has_prior_reaction = np.random.choice([0, 1], n_samples, p=[0.80, 0.20])
#     severity_score = ((ages > 65).astype(int) * 1.5 + (dosages > 100).astype(int) * 1.2 + (np.isin(adverse_events, ["Chest Pain", "Anaphylaxis", "Cardiac Arrest", "Liver Failure", "Dyspnea"])).astype(int) * 2.5 + has_comorbidity * 1.0 + has_prior_reaction * 0.8 + np.random.normal(0, 1, n_samples))
#     severity = pd.cut(severity_score, bins=[-np.inf, 2.5, 5.0, np.inf], labels=["Mild", "Moderate", "Severe"])
#     notes = [f"{ages[i]} year old {genders[i].lower()} patient reported {adverse_events[i].lower()} approximately {time_to_onset[i]} days after taking {dosages[i]}mg of {drug_names[i]} via {routes[i].lower()} route. {'Patient has pre-existing conditions.' if has_comorbidity[i] else ''} {'Patient had prior drug reactions.' if has_prior_reaction[i] else ''}" for i in range(n_samples)]
#     df = pd.DataFrame({"report_id": range(1, n_samples + 1), "age": ages, "gender": genders, "weight_kg": weights, "drug_name": drug_names, "dosage_mg": dosages, "route": routes, "adverse_event": adverse_events, "time_to_onset_days": time_to_onset, "num_concomitant_drugs": num_concomitant_drugs, "symptom_count": symptom_count, "has_comorbidity": has_comorbidity, "has_prior_reaction": has_prior_reaction, "severity": severity, "clinical_note": notes})
#     if save:
#         path = DATA_DIR / "faers_synthetic.csv"
#         df.to_csv(path, index=False)
#         logger.info(f"Saved synthetic data to {path}")
#     logger.info(f"Severity distribution:\n{df['severity'].value_counts()}")
#     return df


# =============================================================================
# REAL FAERS DATA LOADER
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Main entry point for loading data.
    Loads and combines ALL quarters of real FAERS data from data/faers/
    """
    return load_real_faers()


def load_all_quarters(real_path: Path, prefix: str) -> pd.DataFrame:
    """
    Finds ALL quarter files for a given prefix and concatenates them.

    Example:
        DEMO24Q1.txt + DEMO24Q2.txt + DEMO24Q3.txt + DEMO24Q4.txt
        -> one combined DataFrame with all quarters

    Parameters:
        real_path: Path to the faers data folder
        prefix: File prefix to search for e.g. "DEMO", "DRUG"

    Returns:
        Combined DataFrame across all quarters
    """
    # sorted() ensures consistent order: Q1 -> Q2 -> Q3 -> Q4
    matches = sorted(glob.glob(str(real_path / f"{prefix}*.txt")))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {prefix}*.txt in {real_path}\n"
            f"Files present: {list(real_path.iterdir())}"
        )

    logger.info(
        f"Found {len(matches)} {prefix} file(s): "
        f"{[Path(m).name for m in matches]}"
    )

    dfs = []
    for file in matches:
        logger.info(f"  Loading {Path(file).name}...")
        
        if prefix == "DRUG":
            # DRUG is 7.8M rows — only load columns we actually use
            df = pd.read_csv(file, sep="$", encoding="latin1", low_memory=False,
                                usecols=["primaryid", "drugname", "dose_amt", "dose_unit", "route"])
        elif prefix == "DEMO":
            # DEMO — load only columns we need including age_cod and wt_cod for unit conversion
            df = pd.read_csv(file, sep="$", encoding="latin1", low_memory=False,
                                usecols=["primaryid", "caseid", "age", "age_cod", "sex", "wt", "wt_cod", "occr_country"])
        elif prefix == "REAC":
            # REAC — only need primaryid and pt (preferred term / reaction name)
            df = pd.read_csv(file, sep="$", encoding="latin1", low_memory=False,
                                usecols=["primaryid", "pt"])
        elif prefix == "OUTC":
            # OUTC — only need primaryid and outcome code
            df = pd.read_csv(file, sep="$", encoding="latin1", low_memory=False,
                                usecols=["primaryid", "outc_cod"])
        else:
            # Fallback — load everything for any other table
            df = pd.read_csv(file, sep="$", encoding="latin1", low_memory=False)

        df.columns = df.columns.str.lower().str.strip()
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"{prefix} total rows after combining all quarters: {len(combined):,}")
    return combined


def load_real_faers() -> pd.DataFrame:
    """
    Loads and joins all four FAERS tables across all available quarters.

    Folder structure expected:
        data/faers/
            DEMO24Q1.txt, DEMO24Q2.txt, DEMO24Q3.txt, DEMO24Q4.txt
            DRUG24Q1.txt, DRUG24Q2.txt, ...
            REAC24Q1.txt, REAC24Q2.txt, ...
            OUTC24Q1.txt, OUTC24Q2.txt, ...

    FAERS tables used:
        DEMO -> patient demographics (age, gender, weight)
        DRUG -> drug name, dosage, route
        REAC -> adverse reactions reported
        OUTC -> outcome severity codes
    """
    real_path = DATA_DIR / "faers"

    if not real_path.exists():
        raise FileNotFoundError(
            f"FAERS data folder not found at: {real_path}\n"
            "Download from https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html "
            "and unzip the ASCII files into data/faers/"
        )

    logger.info(f"Loading real FAERS data from {real_path}...")

    # ── Load and combine all quarters for each table ──────────────────────────
    logger.info("\nStep 1/4 - Loading DEMO (demographics)...")
    demo = load_all_quarters(real_path, "DEMO")

    logger.info("\nStep 2/4 - Loading DRUG (drug info)... (largest file, may take a moment)")
    drug = load_all_quarters(real_path, "DRUG")

    logger.info("\nStep 3/4 - Loading REAC (adverse reactions)...")
    reac = load_all_quarters(real_path, "REAC")

    logger.info("\nStep 4/4 - Loading OUTC (outcomes)...")
    outc = load_all_quarters(real_path, "OUTC")

    # ── Fix age units — convert everything to years ───────────────────────────
    # age_cod tells us what unit age is stored in:
    # YR=years, DEC=decades, MON=months, WK=weeks, DY=days, HR=hours
    logger.info("\nFixing age units (converting all to years)...")
    age_unit_map = {
        "YR":  1,
        "DEC": 10,
        "MON": 1 / 12,
        "WK":  1 / 52,
        "DY":  1 / 365,
        "HR":  1 / 8760
    }
    demo["age_cod"] = demo["age_cod"].str.strip().str.upper()
    demo["age"] = pd.to_numeric(demo["age"], errors="coerce")
    demo["age"] = demo["age"] * demo["age_cod"].map(age_unit_map).fillna(1)

    # ── Fix weight units — convert everything to kg ───────────────────────────
    # wt_cod tells us what unit weight is stored in: KG or LBS
    logger.info("Fixing weight units (converting all to kg)...")
    demo["wt_cod"] = demo["wt_cod"].str.strip().str.upper()
    demo["wt"] = pd.to_numeric(demo["wt"], errors="coerce")
    demo["wt"] = demo.apply(
        lambda r: r["wt"] * 0.453592 if r["wt_cod"] == "LBS" else r["wt"],
        axis=1
    )

    # ── Select and rename DEMO columns ────────────────────────────────────────
    demo = demo[["primaryid", "caseid", "age", "sex", "wt", "occr_country"]].copy()
    demo.columns = ["report_id", "case_id", "age", "gender", "weight_kg", "country"]

    # ── Process DRUG table ────────────────────────────────────────────────────
    logger.info("\nProcessing DRUG table...")
    drug = drug[["primaryid", "drugname", "dose_amt", "dose_unit", "route"]].copy()
    drug.columns = ["report_id", "drug_name", "dosage_amt", "dosage_unit", "route"]
    # Keep one drug row per report (primary drug)
    drug = drug.groupby("report_id").first().reset_index()
    # Convert dosage to numeric
    drug["dosage_mg"] = pd.to_numeric(drug["dosage_amt"], errors="coerce")

    # ── Process REAC table ────────────────────────────────────────────────────
    logger.info("Processing REAC table...")
    reac = reac[["primaryid", "pt"]].copy()
    reac.columns = ["report_id", "adverse_event"]
    # Keep one reaction per report
    reac = reac.groupby("report_id").first().reset_index()

    # ── Process OUTC table ────────────────────────────────────────────────────
    logger.info("Processing OUTC table...")
    outc = outc[["primaryid", "outc_cod"]].copy()
    outc.columns = ["report_id", "outcome_code"]

    # Map FDA outcome codes to severity labels
    # DE = Death              -> Severe
    # LT = Life Threatening   -> Severe
    # HO = Hospitalization    -> Severe
    # DS = Disability         -> Moderate
    # CA = Congenital Anomaly -> Moderate
    # OT = Other              -> Mild
    # RI = Required Intervention -> Mild
    severity_map = {
        "DE": "Severe",
        "LT": "Severe",
        "HO": "Severe",
        "DS": "Moderate",
        "CA": "Moderate",
        "OT": "Mild",
        "RI": "Mild"
    }
    outc["severity"] = outc["outcome_code"].map(severity_map).fillna("Mild")

    # Keep worst outcome per report (Severe > Moderate > Mild)
    severity_order = {"Severe": 2, "Moderate": 1, "Mild": 0}
    outc["severity_rank"] = outc["severity"].map(severity_order)
    outc = outc.sort_values("severity_rank", ascending=False)
    outc = outc.groupby("report_id").first().reset_index()

    # ── Join all tables ───────────────────────────────────────────────────────
    logger.info("\nJoining all tables...")
    df = demo.merge(
        drug[["report_id", "drug_name", "dosage_mg", "route"]],
        on="report_id", how="left"
    )
    df = df.merge(reac, on="report_id", how="left")
    df = df.merge(outc[["report_id", "severity"]], on="report_id", how="left")

    # ── Post-join cleanup ─────────────────────────────────────────────────────
    # Standardise gender codes to match feature engineering expectations
    gender_map = {"M": "Male", "F": "Female", "UNK": "Unknown", "NS": "Unknown"}
    df["gender"] = df["gender"].map(gender_map).fillna("Unknown")

    # Standardise route values
    route_map = {
        "oral":         "Oral",
        "intravenous":  "Intravenous",
        "subcutaneous": "Subcutaneous",
        "topical":      "Topical"
    }
    df["route"] = df["route"].str.lower().str.strip().map(route_map).fillna("Unknown")

    # Drop rows with no severity label (no outcome filed for this report)
    df = df.dropna(subset=["severity"])

    # ── Add placeholder columns not available in raw FAERS ───────────────────
    # These exist in synthetic data but not in raw FAERS tables.
    # Feature engineering handles them with defaults via clean_data().
    df["num_concomitant_drugs"] = 0   # Would need full DRUG count per report
    df["symptom_count"] = 1           # Would need full REAC count per report
    df["has_comorbidity"] = 0         # Would need INDI table
    df["has_prior_reaction"] = 0      # Would need RPSR table

    # Reconstruct a basic clinical note from available fields
    df["clinical_note"] = (
        df["age"].fillna("unknown").astype(str) + " year old " +
        df["gender"].str.lower() + " patient reported " +
        df["adverse_event"].fillna("adverse event").str.lower()
    )

    # Remove duplicates across quarters (same report appearing in multiple quarters)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["report_id"])
    logger.info(f"Removed {initial_len - len(df):,} duplicate report IDs across quarters")

    logger.info(f"\nFinal dataset: {len(df):,} records")
    logger.info(f"Severity distribution:\n{df['severity'].value_counts()}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSeverity counts:\n{df['severity'].value_counts()}")
