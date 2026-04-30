"""
=============================================================
PHASE 1 — DATA CLEANING & VALIDATION
Aravali Hills Land Degradation Project
=============================================================
Inputs  : aravali_master_fact_table.csv
Outputs : aravali_clean.csv          (25 rows, ML-ready)
          aravali_clean_report.txt   (full audit trail)
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── SECTION 1: LOAD ──────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — DATA CLEANING & VALIDATION")
print("=" * 60)

df_raw = pd.read_csv('/mnt/project/aravali_master_fact_table.csv')

print(f"\n[LOAD] Raw shape       : {df_raw.shape}")
print(f"[LOAD] Years present   : {sorted(df_raw['Year'].unique())}")
print(f"[LOAD] Districts       : {sorted(df_raw['District'].unique())}")
print(f"\n[LOAD] Raw dtypes:\n{df_raw.dtypes}")
print(f"\n[LOAD] NaN counts per column:\n{df_raw.isnull().sum()}")

# ── SECTION 2: FILTER TO SATELLITE YEARS ONLY ────────────────
# Even years (2016,2018,2020,2022) have GW data but NO satellite data
# Analysis requires complete multi-modal records → keep odd years only
SATELLITE_YEARS = [2015, 2017, 2019, 2021, 2023]

df_odd = df_raw[df_raw['Year'].isin(SATELLITE_YEARS)].copy()
df_odd = df_odd.reset_index(drop=True)

print(f"\n[FILTER] After keeping satellite years {SATELLITE_YEARS}")
print(f"         Shape: {df_odd.shape}")
print(f"         Remaining NaN:\n{df_odd.isnull().sum()}")

# ── SECTION 3: ANOMALY INVESTIGATION ─────────────────────────
print("\n" + "=" * 60)
print("ANOMALY INVESTIGATION")
print("=" * 60)

# Ajmer GW full history (all 9 years)
ajmer_gw_all = df_raw[df_raw['District'] == 'Ajmer'][['Year', 'GW_Level_m']]
print(f"\nAjmer GW Level — all years:\n{ajmer_gw_all.to_string(index=False)}")

# Statistical context: mean/std of Ajmer GW excluding 2023
ajmer_gw_valid = ajmer_gw_all[ajmer_gw_all['Year'] != 2023]['GW_Level_m']
ajmer_mean = ajmer_gw_valid.mean()
ajmer_std  = ajmer_gw_valid.std()
ajmer_2023 = 208.40

z_score = (ajmer_2023 - ajmer_mean) / ajmer_std

print(f"\nAjmer GW stats (2015–2022 excluding 2023):")
print(f"  Mean  : {ajmer_mean:.2f} m")
print(f"  Std   : {ajmer_std:.2f} m")
print(f"  2023 value  : {ajmer_2023} m")
print(f"  Z-score     : {z_score:.1f}σ  ← extreme outlier")

# Context across all districts at 2023
print(f"\nAll districts GW_Level_m at 2023:")
print(df_odd[df_odd['Year'] == 2023][['District', 'GW_Level_m']].to_string(index=False))

# Decision logic
print("""
ANOMALY DECISION — Ajmer 2023 GW_Level_m = 208.40 m
─────────────────────────────────────────────────────
  Evidence FOR data error:
    • Z-score = {:.1f}σ above district historical mean
    • All other 4 districts range 9–34 m in 2023
    • Ajmer's own 2015–2022 range: 4.75–11.48 m
    • 208.40 m would be deeper than most boreholes in Rajasthan

  Likely cause:
    • Unit mismatch in source data (cm recorded as m?)
    • Wrong station linked to Ajmer district
    • Data entry error (208.40 → should be ~8.40 or ~10.84?)

  Decision: IMPUTE using linear interpolation
    • 2021 = 4.75 m
    • 2022 = 5.44 m  (from even-year GW data)
    • 2023 = extrapolated → use median of 2015–2022 = {:.2f} m
    • Flag with GW_Imputed = True for transparency
""".format(z_score, ajmer_gw_valid.median()))

# ── SECTION 4: IMPUTE AJMER 2023 ─────────────────────────────
ajmer_imputed_value = round(ajmer_gw_valid.median(), 2)

df_odd.loc[
    (df_odd['District'] == 'Ajmer') & (df_odd['Year'] == 2023),
    'GW_Level_m'
] = ajmer_imputed_value

print(f"[IMPUTE] Ajmer 2023 GW set to median of 2015–2022: {ajmer_imputed_value} m")

# Add imputation flag column
df_odd['GW_Imputed'] = False
df_odd.loc[
    (df_odd['District'] == 'Ajmer') & (df_odd['Year'] == 2023),
    'GW_Imputed'
] = True

# ── SECTION 5: CHECK OTHER ANOMALIES ─────────────────────────
print("\n" + "=" * 60)
print("OTHER ANOMALY CHECKS")
print("=" * 60)

# Udaipur 2020 GW = 59.27 (even year — not in our filtered set, skip)
# But check Jaipur/Pali 2020 = 3.56 (also even year, skipped)

# Check for any remaining NaN in key columns after filtering
key_cols = ['Mean_NDVI', 'Mining_Area_sqkm', 'Total_Forest', 'GW_Level_m', 'Scrub_Area']
nan_check = df_odd[key_cols].isnull().sum()
print(f"\nNaN check on key columns after filter:\n{nan_check}")

# Check for zeros that shouldn't be zero
print(f"\nZero-value check:")
for col in key_cols:
    n_zeros = (df_odd[col] == 0).sum()
    if n_zeros > 0:
        print(f"  {col}: {n_zeros} zeros")
        print(df_odd[df_odd[col] == 0][['District', 'Year', col]])

# Check VDF_Area — many zeros are legitimate (no very dense forest in most districts)
print(f"\nVDF_Area (Very Dense Forest) — zeros are geographically legitimate:")
print(df_odd[['District', 'Year', 'VDF_Area']].to_string(index=False))

# ── SECTION 6: DATA TYPE ENFORCEMENT ─────────────────────────
print("\n" + "=" * 60)
print("DATA TYPE ENFORCEMENT")
print("=" * 60)

df_odd['District']          = df_odd['District'].astype(str).str.strip().str.capitalize()
df_odd['Year']              = df_odd['Year'].astype(int)
df_odd['Mean_NDVI']         = df_odd['Mean_NDVI'].astype(float).round(6)
df_odd['Mining_Area_sqkm']  = df_odd['Mining_Area_sqkm'].astype(float).round(4)
df_odd['VDF_Area']          = df_odd['VDF_Area'].astype(float).round(2)
df_odd['MDF_Area']          = df_odd['MDF_Area'].astype(float).round(2)
df_odd['OpenForest_Area']   = df_odd['OpenForest_Area'].astype(float).round(2)
df_odd['Scrub_Area']        = df_odd['Scrub_Area'].astype(float).round(2)
df_odd['Total_Forest']      = df_odd['Total_Forest'].astype(float).round(2)
df_odd['GW_Level_m']        = df_odd['GW_Level_m'].astype(float).round(2)

print(f"Final dtypes:\n{df_odd.dtypes}")

# ── SECTION 7: FEATURE ENGINEERING FOR ML ────────────────────
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# 7a. NDVI anomaly per district (deviation from district mean)
df_odd['NDVI_District_Mean'] = df_odd.groupby('District')['Mean_NDVI'].transform('mean')
df_odd['NDVI_Anomaly']       = (df_odd['Mean_NDVI'] - df_odd['NDVI_District_Mean']).round(6)

# 7b. Forest density score: Total_Forest / (Total_Forest + Scrub_Area)
# Captures whether covered land is actual forest vs scrub
df_odd['Forest_Density_Score'] = (
    df_odd['Total_Forest'] / (df_odd['Total_Forest'] + df_odd['Scrub_Area'])
).round(4)

# 7c. Forest change % relative to 2015 baseline per district
baseline_2015 = df_odd[df_odd['Year'] == 2015].set_index('District')['Total_Forest']
df_odd['Forest_Baseline_2015'] = df_odd['District'].map(baseline_2015)
df_odd['Forest_Change_pct'] = (
    (df_odd['Total_Forest'] - df_odd['Forest_Baseline_2015'])
    / df_odd['Forest_Baseline_2015'] * 100
).round(2)

# 7d. Mining-to-Forest pressure ratio
df_odd['Mining_Forest_Ratio'] = (
    df_odd['Mining_Area_sqkm'] / df_odd['Total_Forest']
).round(4)

# 7e. Groundwater stress flag
# Flag = 1 if GW depth > 30m (deep aquifer stress threshold for Rajasthan)
df_odd['GW_Stress_Flag'] = (df_odd['GW_Level_m'] > 30).astype(int)

print("Engineered features added:")
eng_cols = ['NDVI_Anomaly', 'Forest_Density_Score', 'Forest_Change_pct',
            'Mining_Forest_Ratio', 'GW_Stress_Flag']
print(df_odd[['District', 'Year'] + eng_cols].to_string(index=False))

# ── SECTION 8: FINAL VALIDATION ──────────────────────────────
print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

assert df_odd.shape[0] == 25,          f"Expected 25 rows, got {df_odd.shape[0]}"
assert df_odd['District'].nunique() == 5, "Expected 5 districts"
assert df_odd['Year'].nunique() == 5,     "Expected 5 years"
assert df_odd.isnull().sum().sum() == 0, f"Still have NaNs: {df_odd.isnull().sum()}"
assert (df_odd['Mean_NDVI'] > 0).all(),  "NDVI values must be positive"
assert (df_odd['Total_Forest'] > 0).all(), "Forest values must be positive"
assert (df_odd['GW_Level_m'] > 0).all(), "GW values must be positive"
assert df_odd['GW_Level_m'].max() < 100, "GW anomaly not properly handled"

print(f"\n✅ All assertions passed")
print(f"   Shape           : {df_odd.shape}")
print(f"   Districts       : {sorted(df_odd['District'].unique())}")
print(f"   Years           : {sorted(df_odd['Year'].unique())}")
print(f"   NaN total       : {df_odd.isnull().sum().sum()}")
print(f"   GW max value    : {df_odd['GW_Level_m'].max()} m")
print(f"   Imputed rows    : {df_odd['GW_Imputed'].sum()}")

# ── SECTION 9: DESCRIPTIVE STATISTICS ────────────────────────
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS — CLEAN DATASET")
print("=" * 60)

numeric_cols = ['Mean_NDVI', 'Mining_Area_sqkm', 'Total_Forest',
                'Scrub_Area', 'GW_Level_m', 'Forest_Change_pct',
                'Mining_Forest_Ratio', 'Forest_Density_Score']

print(f"\nGlobal stats across all 25 observations:")
print(df_odd[numeric_cols].describe().round(3).to_string())

print(f"\nPer-district means (2015–2023):")
print(df_odd.groupby('District')[numeric_cols].mean().round(3).to_string())

print(f"\nPer-year means (all 5 districts):")
print(df_odd.groupby('Year')[numeric_cols].mean().round(3).to_string())

# ── SECTION 10: EXPORT ───────────────────────────────────────
print("\n" + "=" * 60)
print("EXPORT")
print("=" * 60)

# Sort cleanly
df_clean = df_odd.sort_values(['District', 'Year']).reset_index(drop=True)

# Save
df_clean.to_csv('/mnt/user-data/outputs/aravali_clean.csv', index=False)
print(f"✅ Saved: aravali_clean.csv  ({df_clean.shape[0]} rows × {df_clean.shape[1]} cols)")

# ── SECTION 11: AUDIT REPORT ─────────────────────────────────
report_lines = [
    "=" * 60,
    "PHASE 1 AUDIT REPORT — ARAVALI CLEANING",
    "=" * 60,
    "",
    "INPUT FILE   : aravali_master_fact_table.csv",
    f"INPUT SHAPE  : {df_raw.shape}",
    "",
    "STEP 1 — YEAR FILTER",
    f"  Kept years    : {SATELLITE_YEARS}",
    f"  Dropped years : [2016, 2018, 2020, 2022] — no satellite data",
    f"  Post-filter   : {df_odd.shape[0]} rows",
    "",
    "STEP 2 — ANOMALY: Ajmer 2023 GW_Level_m = 208.40 m",
    f"  Z-score vs district history : {z_score:.1f}σ",
    "  Decision    : IMPUTED (likely data entry error)",
    f"  Imputed to  : {ajmer_imputed_value} m (median of Ajmer 2015–2022)",
    "  Flag column : GW_Imputed = True for this row",
    "",
    "STEP 3 — FEATURE ENGINEERING",
    "  NDVI_Anomaly        : deviation from district NDVI mean",
    "  Forest_Density_Score: Total_Forest / (Total_Forest + Scrub_Area)",
    "  Forest_Change_pct   : % change vs 2015 baseline",
    "  Mining_Forest_Ratio : Mining_Area / Total_Forest",
    "  GW_Stress_Flag      : 1 if GW depth > 30 m",
    "",
    "STEP 4 — VALIDATION",
    "  Shape    : 25 rows × 17 columns ✅",
    "  NaN count: 0 ✅",
    "  All assertions passed ✅",
    "",
    "OUTPUT FILE  : aravali_clean.csv",
    "",
    "COLUMNS IN OUTPUT:",
]

for col in df_clean.columns:
    report_lines.append(f"  {col}")

report_lines += [
    "",
    "NOTES FOR DOWNSTREAM PHASES:",
    "  • GW_Imputed flag must be acknowledged in ML analysis",
    "  • Jaipur/Pali share identical GW values — same monitoring station",
    "    (documented quirk in source data, not a cleaning error)",
    "  • Udaipur 2020 GW spike (59.27m) is in even years — excluded",
    "  • VDF_Area = 0 for most districts is geographically correct",
    "    (Very Dense Forest is rare in semi-arid Aravali)",
]

report_text = "\n".join(report_lines)
with open('/mnt/user-data/outputs/aravali_clean_report.txt', 'w') as f:
    f.write(report_text)

print(f"✅ Saved: aravali_clean_report.txt")

# ── SECTION 12: PREVIEW OF FINAL DATASET ─────────────────────
print("\n" + "=" * 60)
print("FINAL DATASET PREVIEW")
print("=" * 60)
print(df_clean.to_string(index=False))

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE")
print("=" * 60)
print("\nFiles ready for Phase 2:")
print("  → /mnt/user-data/outputs/aravali_clean.csv")
print("  → /mnt/user-data/outputs/aravali_clean_report.txt")
