# aravali-land-degradation
Land degradation analysis of Aravali Hills, Rajasthan using satellite data, Mann-Kendall, regression, clustering, and a composite degradation index.
# 🏔 Aravali Hills — Land Degradation Analysis

> **DSC413 Course Project — Deliverable 1 & 2**  
> Rajasthan, India · 5 Districts · 2015–2023

---

## Project Overview

This project investigates land degradation across five districts 
in the Aravali Hills of Rajasthan — Ajmer, Bhilwara, Jaipur, 
Pali, and Udaipur — using a multi-indicator data-driven approach 
spanning 2015 to 2023.

## Methods Applied

| Method | Purpose |
|--------|---------|
| Mann-Kendall Trend Test | Statistical trend detection per district per metric |
| OLS Linear Regression | Quantify relationships between degradation indicators |
| K-Means Clustering + PCA | Group districts by degradation profile |
| Composite Degradation Index | Single unified risk score per district per year |

## Data Sources

- **Google Earth Engine** — Landsat 8 NDVI + Dynamic World LULC
- **Forest Survey of India (FSI)** — Biennial forest cover tables
- **Central Ground Water Board (CGWB)** — Annual GW depth data

## Key Findings

1. **Jaipur** is critically degraded (DI = 0.693) — chronic 
   groundwater stress at 30–35m depth across all years
2. **Udaipur** is the ecological anchor — lowest DI (0.095), 
   confirmed declining mining trend (MK τ = −1.0, p = 0.027)
3. **Ajmer** shows confirmed forest recovery (MK τ = +1.0, p = 0.027)
4. **Forest cover** is the strongest predictor of NDVI health 
   (r = +0.683, R² = 0.467, p = 0.0002)

## Repository Structure
├── report/          → Combined D1+D2 Word report

├── dashboard/       → Interactive HTML dashboard (open in browser)

├── data/processed/  → All cleaned and analysed CSV outputs

├── scripts/         → Python analysis scripts (Phase 1–2d)

├── notebooks/       → Original D1 Jupyter notebook

├── figures/         → All 14 publication-quality figures

└── requirements.txt → Python dependencies

## Interactive Dashboard

Open `dashboard/aravali_dashboard.html` in any browser.  
No installation required — all data is embedded.

**Three tabs:**
- **District Explorer** — time series + Mann-Kendall trends
- **Correlation Explorer** — live scatter plots + regression
- **Degradation Dashboard** — DI heatmap + risk ranking table

## Report

See `report/aravali_report.docx` for the full combined 
Deliverable 1 + Deliverable 2 academic report.
