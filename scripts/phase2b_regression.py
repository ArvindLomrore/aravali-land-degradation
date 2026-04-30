"""
=============================================================
PHASE 2b — CORRELATION & REGRESSION ANALYSIS
Aravali Hills Land Degradation Project
=============================================================
OLS regression implemented from scratch — no scipy/sklearn.
Inputs  : aravali_clean.csv  |  mk_results.csv
Outputs : regression_results.csv
          fig_correlation_heatmap.png
          fig_regression_panel.png
          fig_regression_within_district.png
          fig_partial_correlations.png
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── LOAD ─────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/outputs/aravali_clean.csv')
mk = pd.read_csv('/mnt/user-data/outputs/mk_results.csv')

DISTRICTS = sorted(df['District'].unique())
DISTRICT_COLORS = {
    'Ajmer'   : '#60a5fa',
    'Bhilwara': '#f472b6',
    'Jaipur'  : '#34d399',
    'Pali'    : '#fb923c',
    'Udaipur' : '#a78bfa',
}
DISTRICT_MARKERS = {
    'Ajmer'   : 'o',
    'Bhilwara': 's',
    'Jaipur'  : '^',
    'Pali'    : 'D',
    'Udaipur' : 'P',
}

print("=" * 64)
print("PHASE 2b — CORRELATION & REGRESSION ANALYSIS")
print("=" * 64)
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")


# ══════════════════════════════════════════════════════════════
# SECTION 1 — MANUAL OLS IMPLEMENTATION
# ══════════════════════════════════════════════════════════════
"""
For simple linear regression  y = β0 + β1·x + ε:

  β1 = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
  β0 = ȳ - β1·x̄

  SS_res = Σ(yi - ŷi)²
  SS_tot = Σ(yi - ȳ)²
  R²     = 1 - SS_res / SS_tot

  SE(β1) = √(SS_res/(n-2)) / √Σ(xi-x̄)²
  t-stat = β1 / SE(β1)
  p-value via t-distribution CDF (manual approximation, df = n-2)
"""

def ols_simple(x, y):
    """
    Manual OLS simple linear regression.
    Returns dict: beta0, beta1, r2, r, pearson_p,
                  t_stat, p_value, se_beta1, n, y_pred
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    x_mean = x.mean()
    y_mean = y.mean()

    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_yy = np.sum((y - y_mean) ** 2)

    # Coefficients
    beta1 = ss_xy / ss_xx if ss_xx != 0 else 0.0
    beta0 = y_mean - beta1 * x_mean

    # Predictions & residuals
    y_pred  = beta0 + beta1 * x
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = ss_yy

    # R²
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Pearson r  (signed)
    r = ss_xy / np.sqrt(ss_xx * ss_yy) if (ss_xx * ss_yy) > 0 else 0.0

    # Standard error of β1
    mse     = ss_res / max(n - 2, 1)
    se_b1   = np.sqrt(mse / ss_xx) if ss_xx != 0 else np.inf

    # t-statistic
    t_stat  = beta1 / se_b1 if se_b1 > 0 else 0.0

    # p-value: two-tailed t-distribution, df = n-2
    # Using a rational approximation for the t-CDF (Abramowitz & Stegun)
    df_val = n - 2

    def t_pvalue_twotailed(t, df):
        """Two-tailed p-value from t-distribution via incomplete beta function."""
        t = abs(t)
        if df <= 0:
            return 1.0
        # Use normal approximation for large df, exact series for small
        if df >= 30:
            # Normal approximation
            def phi(z):
                t_ = 1.0 / (1.0 + 0.2316419 * abs(z))
                poly = t_ * (0.319381530
                       + t_ * (-0.356563782
                       + t_ * (1.781477937
                       + t_ * (-1.821255978
                       + t_ *  1.330274429))))
                return 1.0 - (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*z**2) * poly
            p = 2.0 * (1.0 - phi(t))
        else:
            # Series expansion for t-distribution CDF
            x_val = df / (df + t * t)
            # Regularised incomplete beta I(x; df/2, 1/2)
            # Using continued fraction for small df
            a = df / 2.0
            b = 0.5
            # Lentz continued fraction
            tiny = 1e-30
            fpmin = tiny
            qab   = a + b
            qap   = a + 1.0
            qam   = a - 1.0
            c     = 1.0
            d     = 1.0 - qab * x_val / qap
            if abs(d) < fpmin: d = fpmin
            d = 1.0 / d
            h = d
            for m in range(1, 200):
                m2 = 2 * m
                # Even step
                aa = m * (b - m) * x_val / ((qam + m2) * (a + m2))
                d  = 1.0 + aa * d
                if abs(d) < fpmin: d = fpmin
                c  = 1.0 + aa / c
                if abs(c) < fpmin: c = fpmin
                d  = 1.0 / d
                h *= d * c
                # Odd step
                aa = -(a + m) * (qab + m) * x_val / ((a + m2) * (qap + m2))
                d  = 1.0 + aa * d
                if abs(d) < fpmin: d = fpmin
                c  = 1.0 + aa / c
                if abs(c) < fpmin: c = fpmin
                d  = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < 1e-10:
                    break
            # Log beta function
            import math
            log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
            bt = np.exp(math.log(x_val) * a + math.log(1.0 - x_val) * b - log_beta)
            ibeta = bt * h / a
            ibeta = max(0.0, min(1.0, ibeta))
            p = ibeta          # = P(T > t) * 2  for two-tailed
        return max(0.0, min(1.0, p))

    p_value = t_pvalue_twotailed(t_stat, df_val)

    # Pearson p-value (same t-test, same result — kept separate for clarity)
    pearson_p = p_value

    return {
        'beta0'    : round(beta0,  6),
        'beta1'    : round(beta1,  6),
        'r'        : round(r,      4),
        'r2'       : round(r2,     4),
        'pearson_p': round(p_value, 4),
        't_stat'   : round(t_stat, 4),
        'se_beta1' : round(se_b1,  6),
        'n'        : n,
        'y_pred'   : y_pred,
    }


def pearson_r(x, y):
    """Pearson r via manual formula."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    return float(np.sum(xm * ym) / denom) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# SECTION 2 — PEARSON CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("PEARSON CORRELATION MATRIX")
print("=" * 64)

CORR_COLS = {
    'Mean_NDVI'         : 'NDVI',
    'Mining_Area_sqkm'  : 'Mining\nArea',
    'Total_Forest'      : 'Total\nForest',
    'GW_Level_m'        : 'GW\nDepth',
    'Scrub_Area'        : 'Scrub\nArea',
    'Forest_Density_Score':'Forest\nDensity',
    'Mining_Forest_Ratio': 'Mining/\nForest',
    'Forest_Change_pct'  : 'Forest\nChange%',
}

col_keys    = list(CORR_COLS.keys())
col_labels  = list(CORR_COLS.values())
n_cols      = len(col_keys)

corr_matrix = np.zeros((n_cols, n_cols))
for i, ci in enumerate(col_keys):
    for j, cj in enumerate(col_keys):
        corr_matrix[i, j] = pearson_r(df[ci].values, df[cj].values)

corr_df = pd.DataFrame(corr_matrix, index=col_labels, columns=col_labels)
print("\nCorrelation matrix (Pearson r):")
print(corr_df.round(3).to_string())

# Key correlations to highlight
print("\nTop absolute correlations (excluding diagonal):")
pairs = []
for i in range(n_cols):
    for j in range(i+1, n_cols):
        pairs.append((abs(corr_matrix[i,j]), corr_matrix[i,j],
                      col_keys[i], col_keys[j]))
pairs.sort(reverse=True)
for abs_r, r, c1, c2 in pairs[:8]:
    print(f"  {c1:<22} × {c2:<22}  r = {r:+.3f}")


# ══════════════════════════════════════════════════════════════
# SECTION 3 — THREE TARGETED REGRESSIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("OLS REGRESSION RESULTS")
print("=" * 64)

REGRESSIONS = [
    {
        'x_col'  : 'Mining_Area_sqkm',
        'y_col'  : 'GW_Level_m',
        'x_label': 'Mining Area (sq km)',
        'y_label': 'Groundwater Depth (m)',
        'title'  : 'Mining Pressure → Groundwater Depletion',
        'hypothesis': 'Hypothesis: Greater mining activity increases groundwater depth (depletion)',
        'color'  : '#f87171',
    },
    {
        'x_col'  : 'Total_Forest',
        'y_col'  : 'Mean_NDVI',
        'x_label': 'Total Forest Area (sq km)',
        'y_label': 'Mean NDVI',
        'title'  : 'Forest Cover → Vegetation Health',
        'hypothesis': 'Hypothesis: More forest cover corresponds to higher NDVI',
        'color'  : '#34d399',
    },
    {
        'x_col'  : 'Mining_Area_sqkm',
        'y_col'  : 'Mean_NDVI',
        'x_label': 'Mining Area (sq km)',
        'y_label': 'Mean NDVI',
        'title'  : 'Mining Pressure → Vegetation Health',
        'hypothesis': 'Hypothesis: Higher mining area suppresses NDVI',
        'color'  : '#fb923c',
    },
]

# Also run per-district regressions for the main pair (Mining → GW)
reg_records = []

for reg in REGRESSIONS:
    x = df[reg['x_col']].values
    y = df[reg['y_col']].values
    res = ols_simple(x, y)

    sig = "✅ SIGNIFICANT" if res['pearson_p'] < 0.05 else "  not significant"
    direction = "positive" if res['beta1'] > 0 else "negative"

    print(f"\n{'─'*64}")
    print(f"  {reg['title']}")
    print(f"  {reg['hypothesis']}")
    print(f"{'─'*64}")
    print(f"  β0 (intercept)  = {res['beta0']:+.4f}")
    print(f"  β1 (slope)      = {res['beta1']:+.6f}")
    print(f"  Pearson r       = {res['r']:+.4f}  ({direction} correlation)")
    print(f"  R²              = {res['r2']:.4f}  ({res['r2']*100:.1f}% variance explained)")
    print(f"  t-statistic     = {res['t_stat']:+.4f}")
    print(f"  p-value         = {res['pearson_p']:.4f}  {sig}")
    print(f"  n               = {res['n']}")

    reg['result'] = res

    reg_records.append({
        'X_Variable'   : reg['x_col'],
        'Y_Variable'   : reg['y_col'],
        'Beta0'        : res['beta0'],
        'Beta1'        : res['beta1'],
        'Pearson_r'    : res['r'],
        'R_squared'    : res['r2'],
        't_statistic'  : res['t_stat'],
        'p_value'      : res['pearson_p'],
        'Significant'  : res['pearson_p'] < 0.05,
        'n'            : res['n'],
    })

# Per-district regressions: Mining → GW (within each district across 5 years)
print(f"\n{'─'*64}")
print("  PER-DISTRICT: Mining Area → GW Depth  (n=5 per district)")
print(f"{'─'*64}")
within_records = []
for district in DISTRICTS:
    sub = df[df['District'] == district]
    x   = sub['Mining_Area_sqkm'].values
    y   = sub['GW_Level_m'].values
    res = ols_simple(x, y)
    sig = "✅ sig" if res['pearson_p'] < 0.05 else "not sig"
    print(f"  {district:<10}  r={res['r']:+.3f}  R²={res['r2']:.3f}  "
          f"β1={res['beta1']:+.4f}  p={res['pearson_p']:.4f}  {sig}")
    within_records.append({
        'District': district,
        'r': res['r'], 'R2': res['r2'],
        'beta0': res['beta0'], 'beta1': res['beta1'],
        'p_value': res['pearson_p'], 'Significant': res['pearson_p'] < 0.05,
        'result': res,
    })

# Per-district: Forest → NDVI
print(f"\n{'─'*64}")
print("  PER-DISTRICT: Total Forest → NDVI  (n=5 per district)")
print(f"{'─'*64}")
within_forest = []
for district in DISTRICTS:
    sub = df[df['District'] == district]
    x   = sub['Total_Forest'].values
    y   = sub['Mean_NDVI'].values
    res = ols_simple(x, y)
    sig = "✅ sig" if res['pearson_p'] < 0.05 else "not sig"
    print(f"  {district:<10}  r={res['r']:+.3f}  R²={res['r2']:.3f}  "
          f"β1={res['beta1']:+.6f}  p={res['pearson_p']:.4f}  {sig}")
    within_forest.append({
        'District': district,
        'r': res['r'], 'R2': res['r2'],
        'beta0': res['beta0'], 'beta1': res['beta1'],
        'p_value': res['pearson_p'], 'Significant': res['pearson_p'] < 0.05,
        'result': res,
    })

# Save regression CSV
pd.DataFrame(reg_records).to_csv(
    '/mnt/user-data/outputs/regression_results.csv', index=False)
print(f"\n✅ Saved: regression_results.csv")


# ══════════════════════════════════════════════════════════════
# VISUALISATION 1 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════
cmap_div = LinearSegmentedColormap.from_list(
    'rg_div',
    ['#b71c1c','#ef5350','#ffcdd2','#fafafa','#c8e6c9','#66bb6a','#1a7a3c'],
    N=256
)

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

im = ax.imshow(corr_matrix, cmap=cmap_div, vmin=-1, vmax=1, aspect='auto')

# Annotations
for i in range(n_cols):
    for j in range(n_cols):
        val = corr_matrix[i, j]
        txt_col = 'white' if abs(val) > 0.55 else '#1a1a1a'
        weight  = 'bold'  if abs(val) > 0.55 and i != j else 'normal'
        size    = 9       if i != j else 8
        ax.text(j, i, f"{val:+.2f}",
                ha='center', va='center',
                color=txt_col, fontsize=size,
                fontweight=weight, fontfamily='monospace')

# Highlight significant off-diagonal cells
for i in range(n_cols):
    for j in range(n_cols):
        if i != j and abs(corr_matrix[i, j]) >= 0.5:
            rect = plt.Rectangle(
                (j - 0.48, i - 0.48), 0.96, 0.96,
                fill=False, edgecolor='#fbbf24', lw=1.8
            )
            ax.add_patch(rect)

ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, color='#f9fafb', fontsize=9.5,
                   fontfamily='monospace', rotation=0, ha='center')
ax.set_yticks(range(n_cols))
ax.set_yticklabels(col_labels, color='#f9fafb', fontsize=9.5,
                   fontfamily='monospace')
ax.tick_params(length=0)

# Grid
for k in range(n_cols + 1):
    ax.axhline(k - 0.5, color='#0d1117', lw=1.5)
    ax.axvline(k - 0.5, color='#0d1117', lw=1.5)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Pearson r", color='#9ca3af', fontsize=10, fontfamily='monospace')
cbar.ax.yaxis.set_tick_params(color='#9ca3af')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9ca3af', fontfamily='monospace')
cbar.outline.set_edgecolor('#374151')

ax.set_title(
    'Pearson Correlation Matrix — Aravali Degradation Indicators\n'
    'All 25 observations (5 districts × 5 years) | Gold border = |r| ≥ 0.50',
    color='#f9fafb', fontsize=12, fontweight='bold',
    fontfamily='monospace', pad=14
)

plt.tight_layout(pad=1.5)
plt.savefig('/mnt/user-data/outputs/fig_correlation_heatmap.png',
            dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: fig_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════
# VISUALISATION 2 — 3-PANEL REGRESSION SCATTER PLOTS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
fig.patch.set_facecolor('#0d1117')

for ax, reg in zip(axes, REGRESSIONS):
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values():
        spine.set_edgecolor('#374151')
    ax.tick_params(colors='#9ca3af', labelsize=8.5)

    res = reg['result']
    x_all = df[reg['x_col']].values
    y_all = df[reg['y_col']].values

    # Plot each district with its own colour & marker
    for district in DISTRICTS:
        sub = df[df['District'] == district]
        ax.scatter(
            sub[reg['x_col']], sub[reg['y_col']],
            c=DISTRICT_COLORS[district],
            marker=DISTRICT_MARKERS[district],
            s=80, zorder=5, alpha=0.92,
            edgecolors='white', linewidths=0.6,
            label=district
        )

    # Global regression line
    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
    y_fit = res['beta0'] + res['beta1'] * x_fit
    ls    = '-'  if res['pearson_p'] < 0.05 else '--'
    lw    = 2.2  if res['pearson_p'] < 0.05 else 1.4
    ax.plot(x_fit, y_fit, color=reg['color'], lw=lw, ls=ls,
            alpha=0.85, zorder=4)

    # Annotation box
    sig_str = "p < 0.05 ✱" if res['pearson_p'] < 0.05 else f"p = {res['pearson_p']:.3f}"
    annotation = (
        f"r  = {res['r']:+.3f}\n"
        f"R² = {res['r2']:.3f}\n"
        f"{sig_str}\n"
        f"y = {res['beta0']:.2f} + {res['beta1']:.4f}x"
    )
    ax.text(0.97, 0.97, annotation,
            transform=ax.transAxes,
            fontsize=8.5, color='#f9fafb',
            ha='right', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#1f2937', edgecolor='#374151',
                      alpha=0.92))

    ax.set_xlabel(reg['x_label'], color='#9ca3af', fontsize=9,
                  fontfamily='monospace')
    ax.set_ylabel(reg['y_label'], color='#9ca3af', fontsize=9,
                  fontfamily='monospace')
    ax.set_title(reg['title'], color='#f9fafb', fontsize=10,
                 fontweight='bold', fontfamily='monospace', pad=8)
    ax.grid(True, color='#21262d', lw=0.6, alpha=0.8)

# Shared legend
handles = [
    mpatches.Patch(facecolor=DISTRICT_COLORS[d],
                   edgecolor='white', label=d)
    for d in DISTRICTS
]
solid  = plt.Line2D([0],[0], color='white', lw=2.2,
                    label='Regression line (sig.)')
dashed = plt.Line2D([0],[0], color='white', lw=1.4, ls='--',
                    label='Regression line (non-sig.)')
handles += [solid, dashed]

fig.legend(handles=handles, loc='lower center', ncol=7,
           bbox_to_anchor=(0.5, 0.01),
           frameon=True, framealpha=0.15,
           facecolor='#1f2937', edgecolor='#374151',
           labelcolor='#e2e8f0', fontsize=9)

fig.suptitle(
    'OLS Regression Analysis — Aravali Environmental Indicators\n'
    'Global pooled regression across all 25 observations',
    color='#f9fafb', fontsize=13, fontweight='bold',
    fontfamily='monospace', y=1.00
)

plt.tight_layout(pad=1.5, rect=[0, 0.08, 1, 1])
plt.savefig('/mnt/user-data/outputs/fig_regression_panel.png',
            dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: fig_regression_panel.png")


# ══════════════════════════════════════════════════════════════
# VISUALISATION 3 — WITHIN-DISTRICT REGRESSION: Mining → GW
# 5 subplots, one per district, with per-district OLS line
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(20, 5.5))
fig.patch.set_facecolor('#0d1117')

for ax, district, rec in zip(axes, DISTRICTS, within_records):
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values():
        spine.set_edgecolor('#374151')
    ax.tick_params(colors='#9ca3af', labelsize=8)

    sub = df[df['District'] == district].sort_values('Year')
    x   = sub['Mining_Area_sqkm'].values
    y   = sub['GW_Level_m'].values
    res = rec['result']
    col = DISTRICT_COLORS[district]

    # Scatter — colour-coded by year
    year_cmap = plt.cm.get_cmap('cool', 5)
    for idx, (xi, yi, yr) in enumerate(
            zip(x, y, sub['Year'].values)):
        ax.scatter(xi, yi,
                   color=year_cmap(idx),
                   s=90, zorder=5,
                   edgecolors='white', linewidths=0.8)
        ax.annotate(str(yr), (xi, yi),
                    textcoords='offset points',
                    xytext=(4, 4),
                    fontsize=7, color='#9ca3af',
                    fontfamily='monospace')

    # Regression line
    x_fit = np.linspace(x.min() - 5, x.max() + 5, 200)
    y_fit = res['beta0'] + res['beta1'] * x_fit
    ls  = '-'  if rec['Significant'] else '--'
    ax.plot(x_fit, y_fit, color=col, lw=2.0, ls=ls, alpha=0.9)

    # Stats annotation
    sig_str = "p<0.05 ✱" if rec['Significant'] else f"p={rec['p_value']:.3f}"
    ax.text(0.97, 0.97,
            f"r  = {rec['r']:+.2f}\nR²= {rec['R2']:.2f}\n{sig_str}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#f9fafb', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1f2937',
                      edgecolor='#374151', alpha=0.9))

    ax.set_title(district, color=col, fontsize=11,
                 fontweight='bold', fontfamily='monospace')
    ax.set_xlabel('Mining Area (sq km)', color='#9ca3af',
                  fontsize=8, fontfamily='monospace')
    ax.set_ylabel('GW Depth (m)', color='#9ca3af',
                  fontsize=8, fontfamily='monospace')
    ax.grid(True, color='#21262d', lw=0.6, alpha=0.8)

fig.suptitle(
    'Within-District Regression: Mining Area → Groundwater Depth\n'
    'Per-district OLS  |  n = 5 time points (2015–2023)',
    color='#f9fafb', fontsize=12, fontweight='bold',
    fontfamily='monospace', y=1.02
)

plt.tight_layout(pad=1.5)
plt.savefig('/mnt/user-data/outputs/fig_regression_within_district.png',
            dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: fig_regression_within_district.png")


# ══════════════════════════════════════════════════════════════
# VISUALISATION 4 — PARTIAL CORRELATION DEEP DIVE
# Bar chart: r values for each predictor vs NDVI & vs GW Depth
# ══════════════════════════════════════════════════════════════
targets     = ['Mean_NDVI', 'GW_Level_m']
target_lbl  = ['NDVI', 'GW Depth']
predictors  = ['Mining_Area_sqkm', 'Total_Forest', 'Scrub_Area',
               'Forest_Density_Score', 'Mining_Forest_Ratio',
               'Forest_Change_pct']
pred_labels = ['Mining\nArea', 'Total\nForest', 'Scrub\nArea',
               'Forest\nDensity', 'Mining/\nForest', 'Forest\nChange%']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#0d1117')

for ax, target, t_lbl in zip(axes, targets, target_lbl):
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values():
        spine.set_edgecolor('#374151')
    ax.tick_params(colors='#9ca3af', labelsize=9)

    r_vals = []
    p_vals = []
    for pred in predictors:
        res = ols_simple(df[pred].values, df[target].values)
        r_vals.append(res['r'])
        p_vals.append(res['pearson_p'])

    colors = ['#1a7a3c' if r > 0 else '#b71c1c' for r in r_vals]
    alphas = [1.0 if p < 0.05 else 0.45 for p in p_vals]

    bars = ax.bar(range(len(predictors)), r_vals, color=colors,
                  alpha=0.85, edgecolor='#0d1117', linewidth=0.8)

    # Apply per-bar alpha manually
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    # Value labels
    for i, (r, p) in enumerate(zip(r_vals, p_vals)):
        y_off = 0.02 if r >= 0 else -0.05
        label = f"{r:+.2f}" + (" ✱" if p < 0.05 else "")
        ax.text(i, r + y_off, label,
                ha='center', va='bottom' if r >= 0 else 'top',
                fontsize=8.5, color='white', fontfamily='monospace',
                fontweight='bold' if p < 0.05 else 'normal')

    ax.axhline(0, color='#9ca3af', lw=1.2, alpha=0.6)
    ax.axhline( 0.5, color='#374151', lw=0.8, ls=':', alpha=0.7)
    ax.axhline(-0.5, color='#374151', lw=0.8, ls=':', alpha=0.7)

    ax.set_xticks(range(len(predictors)))
    ax.set_xticklabels(pred_labels, color='#e2e8f0',
                       fontfamily='monospace', fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(f"Pearson r  (predictor → {t_lbl})",
                  color='#9ca3af', fontfamily='monospace', fontsize=9)
    ax.set_title(f"Predictors of {t_lbl}",
                 color='#f9fafb', fontsize=11, fontweight='bold',
                 fontfamily='monospace', pad=10)
    ax.grid(axis='y', color='#21262d', lw=0.6, alpha=0.8)

    # Legend
    sig_patch   = mpatches.Patch(color='white', alpha=1.0,
                                  label='Significant (p<0.05) ✱')
    insig_patch = mpatches.Patch(color='white', alpha=0.45,
                                  label='Non-significant')
    ax.legend(handles=[sig_patch, insig_patch],
              loc='lower right', frameon=True, framealpha=0.2,
              facecolor='#1f2937', edgecolor='#374151',
              labelcolor='#e2e8f0', fontsize=8)

fig.suptitle(
    'Predictor Correlations with NDVI and Groundwater Depth\n'
    'Green = positive r  |  Red = negative r  |  ✱ = p < 0.05',
    color='#f9fafb', fontsize=12, fontweight='bold',
    fontfamily='monospace', y=1.02
)

plt.tight_layout(pad=2.0)
plt.savefig('/mnt/user-data/outputs/fig_partial_correlations.png',
            dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: fig_partial_correlations.png")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — NARRATIVE INTERPRETATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("KEY INSIGHTS — CORRELATION & REGRESSION")
print("=" * 64)

# Pull key values
r_mining_gw     = pearson_r(df['Mining_Area_sqkm'].values, df['GW_Level_m'].values)
r_forest_ndvi   = pearson_r(df['Total_Forest'].values, df['Mean_NDVI'].values)
r_mining_ndvi   = pearson_r(df['Mining_Area_sqkm'].values, df['Mean_NDVI'].values)
r_mfr_ndvi      = pearson_r(df['Mining_Forest_Ratio'].values, df['Mean_NDVI'].values)
r_fd_ndvi       = pearson_r(df['Forest_Density_Score'].values, df['Mean_NDVI'].values)

print(f"""
GLOBAL POOLED ANALYSIS (n=25):
  Mining Area  → GW Depth   : r = {r_mining_gw:+.3f}
  Total Forest → NDVI       : r = {r_forest_ndvi:+.3f}
  Mining Area  → NDVI       : r = {r_mining_ndvi:+.3f}
  Mining/Forest ratio → NDVI: r = {r_mfr_ndvi:+.3f}
  Forest Density  → NDVI    : r = {r_fd_ndvi:+.3f}

ECOLOGICAL INTERPRETATION:

  1. Mining ↔ GW: r = {r_mining_gw:+.3f}
     Weak positive correlation at pooled level. The global
     signal is diluted because Jaipur & Pali have deep GW
     (30-34m) driven by urban extraction, NOT mining.
     Per-district analysis is more revealing — Bhilwara
     shows the strongest within-district mining-GW link.

  2. Forest → NDVI: r = {r_forest_ndvi:+.3f}
     Strong positive correlation confirms that forest cover
     is a primary driver of vegetation health signal. Udaipur
     dominates this relationship (3120 sq km forest, highest NDVI).

  3. Mining → NDVI: r = {r_mining_ndvi:+.3f}
     Negative correlation — mining suppresses vegetation health.
     The signal is moderate because large-mining districts
     (Bhilwara) partially recovered NDVI after 2015 spike.

  4. Mining/Forest Ratio → NDVI: r = {r_mfr_ndvi:+.3f}
     Strongest negative predictor of NDVI. The RATIO
     captures exploitation pressure better than raw mining
     area — high mining relative to available forest is
     the most ecologically damaging configuration.

  IMPORTANT CAVEAT:
     Pooled analysis mixes between-district and within-district
     variance. Udaipur's large forest area creates a strong
     leverage effect on Forest → NDVI. Per-district regressions
     (Fig 3) reveal heterogeneous responses.
""")

print("=" * 64)
print("PHASE 2b COMPLETE")
print("=" * 64)
print("\nOutputs:")
print("  → regression_results.csv")
print("  → fig_correlation_heatmap.png")
print("  → fig_regression_panel.png")
print("  → fig_regression_within_district.png")
print("  → fig_partial_correlations.png")
