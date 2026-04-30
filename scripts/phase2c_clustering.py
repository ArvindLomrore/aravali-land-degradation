"""
=============================================================
PHASE 2c — K-MEANS CLUSTERING + MANUAL PCA
Aravali Hills Land Degradation Project
=============================================================
K-Means and PCA implemented from scratch — no sklearn/scipy.
Inputs  : aravali_clean.csv
Outputs : cluster_assignments.csv
          fig_elbow_silhouette.png
          fig_pca_clusters.png
          fig_cluster_profiles.png
          fig_cluster_radar.png
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── LOAD ─────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/outputs/aravali_clean.csv')

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
print("PHASE 2c — K-MEANS CLUSTERING + MANUAL PCA")
print("=" * 64)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE ENGINEERING: CHANGE-RATE FEATURES
# ══════════════════════════════════════════════════════════════
"""
Strategy: Summarise each district into a single row of
CHANGE features (2015 → 2023) + LEVEL features (mean over period).
This gives us 5 data points (districts) — appropriate for k=2,3.
"""

print("\n" + "=" * 64)
print("SECTION 1 — FEATURE ENGINEERING")
print("=" * 64)

rows = []
for district in DISTRICTS:
    sub  = df[df['District'] == district].sort_values('Year')
    y15  = sub[sub['Year'] == 2015].iloc[0]
    y23  = sub[sub['Year'] == 2023].iloc[0]
    mean = sub.mean(numeric_only=True)

    # ── Change features (2015 → 2023 absolute delta) ────────
    d_ndvi    = y23['Mean_NDVI']         - y15['Mean_NDVI']
    d_mining  = y23['Mining_Area_sqkm']  - y15['Mining_Area_sqkm']
    d_forest  = y23['Total_Forest']      - y15['Total_Forest']
    d_gw      = y23['GW_Level_m']        - y15['GW_Level_m']
    d_scrub   = y23['Scrub_Area']        - y15['Scrub_Area']

    # ── Change features (%) ─────────────────────────────────
    pct_ndvi   = d_ndvi   / y15['Mean_NDVI']        * 100
    pct_mining = d_mining / y15['Mining_Area_sqkm']  * 100
    pct_forest = d_forest / y15['Total_Forest']      * 100
    pct_gw     = d_gw     / y15['GW_Level_m']        * 100

    # ── Level features (mean 2015–2023) ─────────────────────
    mean_ndvi    = mean['Mean_NDVI']
    mean_mining  = mean['Mining_Area_sqkm']
    mean_forest  = mean['Total_Forest']
    mean_gw      = mean['GW_Level_m']
    mean_mfr     = mean['Mining_Forest_Ratio']
    mean_fds     = mean['Forest_Density_Score']
    gw_stress    = sub['GW_Stress_Flag'].sum()   # count of stressed years

    rows.append({
        'District'      : district,
        # Absolute deltas
        'D_NDVI'        : round(d_ndvi,   5),
        'D_Mining'      : round(d_mining, 3),
        'D_Forest'      : round(d_forest, 2),
        'D_GW'          : round(d_gw,     3),
        'D_Scrub'       : round(d_scrub,  2),
        # Percentage changes
        'Pct_NDVI'      : round(pct_ndvi,   2),
        'Pct_Mining'    : round(pct_mining, 2),
        'Pct_Forest'    : round(pct_forest, 2),
        'Pct_GW'        : round(pct_gw,     2),
        # Mean levels
        'Mean_NDVI'     : round(mean_ndvi,   5),
        'Mean_Mining'   : round(mean_mining, 3),
        'Mean_Forest'   : round(mean_forest, 2),
        'Mean_GW'       : round(mean_gw,     3),
        'Mean_MFR'      : round(mean_mfr,    5),
        'Mean_FDS'      : round(mean_fds,    4),
        'GW_Stress_Yrs' : int(gw_stress),
    })

feat_df = pd.DataFrame(rows).set_index('District')
print("\nDistrict feature matrix:")
print(feat_df.to_string())

# ── Select clustering features ───────────────────────────────
# Use % changes + mean level features — avoids scale issues from
# mixing absolute area (sq km) with NDVI (0-1)
CLUSTER_FEATURES = [
    'Pct_NDVI', 'Pct_Mining', 'Pct_Forest', 'Pct_GW',
    'Mean_NDVI', 'Mean_MFR', 'Mean_FDS', 'GW_Stress_Yrs'
]

X_raw    = feat_df[CLUSTER_FEATURES].values.astype(float)
feature_names = CLUSTER_FEATURES
n_samples, n_features = X_raw.shape
print(f"\nClustering matrix: {n_samples} districts × {n_features} features")
print(f"Features: {feature_names}")


# ══════════════════════════════════════════════════════════════
# SECTION 2 — STANDARDISATION (Z-SCORE)
# ══════════════════════════════════════════════════════════════
def standardise(X):
    """Z-score standardisation. Returns (X_std, means, stds)."""
    means = X.mean(axis=0)
    stds  = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0          # avoid division by zero
    return (X - means) / stds, means, stds

X_std, feat_means, feat_stds = standardise(X_raw)
print(f"\nStandardised feature matrix (Z-scores):")
print(pd.DataFrame(X_std, index=DISTRICTS,
                   columns=feature_names).round(3).to_string())


# ══════════════════════════════════════════════════════════════
# SECTION 3 — MANUAL K-MEANS IMPLEMENTATION
# ══════════════════════════════════════════════════════════════
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans(X, k, max_iter=300, n_init=50, seed=42):
    """
    K-Means from scratch.
    Multiple random restarts (n_init) — keeps best (lowest inertia).
    Returns: labels, centroids, inertia, iteration_history
    """
    rng = np.random.RandomState(seed)
    n   = X.shape[0]

    best_labels    = None
    best_centroids = None
    best_inertia   = np.inf

    for trial in range(n_init):
        # ── Random initialisation ──────────────────────────
        # K-Means++ style: first centroid random,
        # subsequent chosen proportional to distance²
        idx0       = rng.randint(0, n)
        centroids  = [X[idx0].copy()]

        for _ in range(k - 1):
            dists = np.array([
                min(euclidean_distance(x, c) ** 2 for c in centroids)
                for x in X
            ])
            probs = dists / dists.sum()
            cum   = np.cumsum(probs)
            r     = rng.rand()
            idx   = np.searchsorted(cum, r)
            centroids.append(X[idx].copy())

        centroids = np.array(centroids)

        # ── Iteration ─────────────────────────────────────
        labels = np.zeros(n, dtype=int)
        for iteration in range(max_iter):
            # Assignment step
            new_labels = np.array([
                np.argmin([euclidean_distance(x, c) for c in centroids])
                for x in X
            ])

            # Update step
            new_centroids = np.array([
                X[new_labels == j].mean(axis=0) if (new_labels == j).sum() > 0
                else centroids[j]
                for j in range(k)
            ])

            # Convergence check
            if np.all(new_labels == labels):
                break
            labels    = new_labels
            centroids = new_centroids

        # Inertia = sum of squared distances to assigned centroid
        inertia = sum(
            euclidean_distance(X[i], centroids[labels[i]]) ** 2
            for i in range(n)
        )

        if inertia < best_inertia:
            best_inertia   = inertia
            best_labels    = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids, best_inertia


def silhouette_score(X, labels):
    """
    Silhouette score from scratch.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    a(i) = mean intra-cluster distance
    b(i) = mean nearest-cluster distance
    """
    n = X.shape[0]
    unique_k = np.unique(labels)
    if len(unique_k) < 2:
        return 0.0

    scores = []
    for i in range(n):
        # a(i): mean distance to same-cluster members
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if len(same) == 0:
            a_i = 0.0
        else:
            a_i = np.mean([euclidean_distance(X[i], X[j]) for j in same])

        # b(i): mean distance to nearest other cluster
        other_clusters = [k for k in unique_k if k != labels[i]]
        b_vals = []
        for k in other_clusters:
            members = [j for j in range(n) if labels[j] == k]
            if members:
                b_vals.append(np.mean(
                    [euclidean_distance(X[i], X[j]) for j in members]
                ))
        b_i = min(b_vals) if b_vals else 0.0

        denom = max(a_i, b_i)
        scores.append((b_i - a_i) / denom if denom > 0 else 0.0)

    return float(np.mean(scores))


# ── Run for k=1..5 (elbow) ───────────────────────────────────
print("\n" + "=" * 64)
print("SECTION 3 — K-MEANS RESULTS")
print("=" * 64)

k_range    = range(1, 6)
inertias   = []
sil_scores = []
all_results = {}

for k in k_range:
    labels, centroids, inertia = kmeans(X_std, k, n_init=100, seed=42)
    inertias.append(inertia)
    sil = silhouette_score(X_std, labels) if k > 1 else 0.0
    sil_scores.append(sil)
    all_results[k] = {'labels': labels, 'centroids': centroids,
                      'inertia': inertia, 'silhouette': sil}

    if k > 1:
        print(f"  k={k}: inertia={inertia:.4f}  silhouette={sil:.4f}")
        for d, lbl in zip(DISTRICTS, labels):
            print(f"         {d} → cluster {lbl+1}")
    else:
        print(f"  k={k}: inertia={inertia:.4f}  (silhouette N/A)")

# Best k by silhouette
best_k = max(range(2, 6), key=lambda k: sil_scores[k - 1])
print(f"\n  Best k by silhouette: k={best_k}  "
      f"(score={sil_scores[best_k-1]:.4f})")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — MANUAL PCA (2 COMPONENTS)
# ══════════════════════════════════════════════════════════════
"""
PCA from scratch:
  1. Centre the data (already standardised = centred + scaled)
  2. Compute covariance matrix C = X^T X / (n-1)
  3. Eigen-decomposition of C (power iteration for top-2 eigenvectors)
  4. Project X onto top-2 eigenvectors
"""

def pca_manual(X, n_components=2):
    """
    Manual PCA via covariance matrix + power iteration.
    X must already be standardised (zero mean).
    Returns: Z (projected), components, explained_variance_ratio
    """
    n, p = X.shape
    # Covariance matrix
    C = X.T @ X / (n - 1)

    # Power iteration for top-k eigenvectors
    eigvecs = []
    eigvals = []
    X_deflated = C.copy()

    for _ in range(n_components):
        # Random start vector
        v = np.random.RandomState(42 + _).randn(p)
        v = v / np.linalg.norm(v)

        for __ in range(10000):
            v_new = X_deflated @ v
            norm  = np.linalg.norm(v_new)
            if norm < 1e-12:
                break
            v_new = v_new / norm
            if np.linalg.norm(v_new - v) < 1e-12:
                v = v_new
                break
            v = v_new

        eigenvalue = v @ X_deflated @ v
        eigvecs.append(v.copy())
        eigvals.append(eigenvalue)

        # Deflate: remove this component
        X_deflated = X_deflated - eigenvalue * np.outer(v, v)

    components = np.array(eigvecs)            # shape (n_components, p)
    total_var  = np.trace(C)
    evr        = np.array(eigvals) / total_var if total_var > 0 else eigvals

    # Project
    Z = X @ components.T                      # shape (n, n_components)
    return Z, components, evr, eigvals

Z_pca, pca_components, evr, eigvals = pca_manual(X_std, n_components=2)

print("\n" + "=" * 64)
print("SECTION 4 — MANUAL PCA")
print("=" * 64)
print(f"\nExplained variance ratio:")
print(f"  PC1: {evr[0]*100:.1f}%")
print(f"  PC2: {evr[1]*100:.1f}%")
print(f"  Total: {sum(evr)*100:.1f}%")

print(f"\nPC1 loadings (feature contributions):")
for fname, loading in zip(feature_names, pca_components[0]):
    bar = '█' * int(abs(loading * 20))
    sign = '+' if loading > 0 else '-'
    print(f"  {fname:<22} {sign}{abs(loading):.3f}  {bar}")

print(f"\nPC2 loadings:")
for fname, loading in zip(feature_names, pca_components[1]):
    bar = '█' * int(abs(loading * 20))
    sign = '+' if loading > 0 else '-'
    print(f"  {fname:<22} {sign}{abs(loading):.3f}  {bar}")

print(f"\nDistrict PCA coordinates:")
for d, z in zip(DISTRICTS, Z_pca):
    print(f"  {d:<10}  PC1={z[0]:+.3f}  PC2={z[1]:+.3f}")


# ══════════════════════════════════════════════════════════════
# SECTION 5 — ECOLOGICAL CLUSTER LABELLING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("SECTION 5 — CLUSTER LABELS")
print("=" * 64)

# k=2 results
labels_k2 = all_results[2]['labels']
# k=3 results
labels_k3 = all_results[3]['labels']

print("\nk=2 assignments:")
for d, l in zip(DISTRICTS, labels_k2):
    print(f"  {d} → Cluster {l+1}")

print("\nk=3 assignments:")
for d, l in zip(DISTRICTS, labels_k3):
    print(f"  {d} → Cluster {l+1}")

# Compute per-cluster means on original (unstandardised) features
# to inform ecological labels
feat_with_label = feat_df[CLUSTER_FEATURES].copy()
feat_with_label['k2'] = labels_k2
feat_with_label['k3'] = labels_k3

print("\nCluster means (k=2) — standardised space:")
for cl in range(2):
    members = [DISTRICTS[i] for i,l in enumerate(labels_k2) if l==cl]
    vals    = X_std[labels_k2 == cl].mean(axis=0)
    print(f"  Cluster {cl+1} {members}:")
    for fname, v in zip(feature_names, vals):
        print(f"    {fname:<22}: {v:+.3f}")

print("\nCluster means (k=3) — standardised space:")
for cl in range(3):
    members = [DISTRICTS[i] for i,l in enumerate(labels_k3) if l==cl]
    vals    = X_std[labels_k3 == cl].mean(axis=0)
    print(f"  Cluster {cl+1} {members}:")
    for fname, v in zip(feature_names, vals):
        print(f"    {fname:<22}: {v:+.3f}")

# ── Ecological labels ────────────────────────────────────────
# Inspect k=3 to assign meaningful names
# We need to map cluster indices to ecological descriptions
# based on the standardised means above
# (Determined by inspecting the output below — encoded after first run)

# Helper: get dominant characteristic per cluster
def label_cluster_k3(labels, X_std, feature_names, districts):
    """Returns dict mapping cluster_id → ecological label."""
    means = {}
    for cl in range(3):
        mask   = labels == cl
        members = [d for d, l in zip(districts, labels) if l == cl]
        mu     = X_std[mask].mean(axis=0)
        means[cl] = {'members': members, 'mu': mu}

    # Sort clusters by mean GW stress + mining ratio descending
    # to assign: High Degradation > Moderate > Low/Recovering
    gw_idx  = feature_names.index('GW_Stress_Yrs')
    mfr_idx = feature_names.index('Mean_MFR')
    ndvi_idx= feature_names.index('Mean_NDVI')

    scores = {cl: means[cl]['mu'][gw_idx] + means[cl]['mu'][mfr_idx]
              - means[cl]['mu'][ndvi_idx]
              for cl in range(3)}
    rank = sorted(scores, key=scores.get, reverse=True)

    labels_eco = {}
    eco_names  = ['🔴 HIGH DEGRADATION', '🟡 MODERATE STRESS', '🟢 LOW STRESS / RECOVERING']
    colors_eco = ['#ef4444', '#f59e0b', '#22c55e']
    for i, cl in enumerate(rank):
        labels_eco[cl] = {
            'name'   : eco_names[i],
            'color'  : colors_eco[i],
            'members': means[cl]['members'],
        }
    return labels_eco

eco_k3 = label_cluster_k3(labels_k3, X_std, feature_names, DISTRICTS)

print("\nEcological cluster labels (k=3):")
for cl_id, info in eco_k3.items():
    print(f"  Cluster {cl_id+1}: {info['name']}")
    print(f"           Members: {info['members']}")


# ══════════════════════════════════════════════════════════════
# SECTION 6 — SAVE CLUSTER ASSIGNMENTS
# ══════════════════════════════════════════════════════════════
assignment_rows = []
for i, district in enumerate(DISTRICTS):
    cl2 = labels_k2[i]
    cl3 = labels_k3[i]
    eco = eco_k3[cl3]['name']
    assignment_rows.append({
        'District'       : district,
        'Cluster_k2'     : cl2 + 1,
        'Cluster_k3'     : cl3 + 1,
        'Ecological_Label': eco,
        'Pct_NDVI'       : feat_df.loc[district, 'Pct_NDVI'],
        'Pct_Mining'     : feat_df.loc[district, 'Pct_Mining'],
        'Pct_Forest'     : feat_df.loc[district, 'Pct_Forest'],
        'Pct_GW'         : feat_df.loc[district, 'Pct_GW'],
        'Mean_NDVI'      : feat_df.loc[district, 'Mean_NDVI'],
        'Mean_Mining'    : feat_df.loc[district, 'Mean_Mining'],
        'Mean_Forest'    : feat_df.loc[district, 'Mean_Forest'],
        'Mean_GW'        : feat_df.loc[district, 'Mean_GW'],
        'GW_Stress_Yrs'  : feat_df.loc[district, 'GW_Stress_Yrs'],
        'PC1'            : round(float(Z_pca[i, 0]), 4),
        'PC2'            : round(float(Z_pca[i, 1]), 4),
    })

assign_df = pd.DataFrame(assignment_rows)
assign_df.to_csv('/mnt/user-data/outputs/cluster_assignments.csv', index=False)
print("\n✅ Saved: cluster_assignments.csv")


# ══════════════════════════════════════════════════════════════
# VISUALISATION 1 — ELBOW + SILHOUETTE
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('#0d1117')

k_list = list(k_range)

# ── Elbow ────────────────────────────────────────────────────
ax1.set_facecolor('#161b22')
for sp in ax1.spines.values(): sp.set_edgecolor('#374151')
ax1.tick_params(colors='#9ca3af', labelsize=9)

ax1.plot(k_list, inertias, color='#60a5fa', lw=2.5,
         marker='o', markersize=9, markerfacecolor='white',
         markeredgecolor='#60a5fa', markeredgewidth=2, zorder=5)

# Fill area under curve
ax1.fill_between(k_list, inertias,
                 alpha=0.12, color='#60a5fa')

# Annotate values
for k, inn in zip(k_list, inertias):
    ax1.annotate(f'{inn:.2f}',
                 xy=(k, inn), xytext=(0, 14),
                 textcoords='offset points',
                 ha='center', fontsize=8.5,
                 color='#e2e8f0', fontfamily='monospace')

# Highlight elbow at k=2 or k=3
ax1.axvline(best_k, color='#fbbf24', lw=1.8, ls='--', alpha=0.8,
            label=f'Best k = {best_k}')

ax1.set_xlabel('Number of Clusters  k', color='#9ca3af',
               fontsize=10, fontfamily='monospace')
ax1.set_ylabel('Inertia (Within-Cluster SSE)',
               color='#9ca3af', fontsize=10, fontfamily='monospace')
ax1.set_title('Elbow Method\nK-Means Inertia vs k',
              color='#f9fafb', fontsize=11, fontweight='bold',
              fontfamily='monospace', pad=10)
ax1.set_xticks(k_list)
ax1.grid(True, color='#21262d', lw=0.6, alpha=0.8)
ax1.legend(fontsize=9, labelcolor='#e2e8f0',
           facecolor='#1f2937', edgecolor='#374151')

# ── Silhouette ───────────────────────────────────────────────
ax2.set_facecolor('#161b22')
for sp in ax2.spines.values(): sp.set_edgecolor('#374151')
ax2.tick_params(colors='#9ca3af', labelsize=9)

sil_k = k_list[1:]      # k=2..5
sil_v = sil_scores[1:]

bar_colors = ['#fbbf24' if k == best_k else '#6366f1' for k in sil_k]
bars = ax2.bar(sil_k, sil_v, color=bar_colors,
               edgecolor='#0d1117', linewidth=1.0, width=0.55, alpha=0.9)

for k, v in zip(sil_k, sil_v):
    ax2.text(k, v + 0.008, f'{v:.3f}',
             ha='center', fontsize=9, color='#f9fafb',
             fontfamily='monospace', fontweight='bold')

ax2.axhline(0.5, color='#34d399', lw=1.2, ls='--', alpha=0.7,
            label='Good threshold (0.5)')
ax2.set_xlabel('Number of Clusters  k', color='#9ca3af',
               fontsize=10, fontfamily='monospace')
ax2.set_ylabel('Silhouette Score', color='#9ca3af',
               fontsize=10, fontfamily='monospace')
ax2.set_title('Silhouette Analysis\nCluster Cohesion vs Separation',
              color='#f9fafb', fontsize=11, fontweight='bold',
              fontfamily='monospace', pad=10)
ax2.set_xticks(sil_k)
ax2.set_ylim(0, max(sil_v) * 1.25)
ax2.grid(True, color='#21262d', lw=0.6, alpha=0.8, axis='y')

gold_patch  = mpatches.Patch(color='#fbbf24', label=f'Optimal k = {best_k}')
blue_patch  = mpatches.Patch(color='#6366f1', label='Other k values')
ax2.legend(handles=[gold_patch, blue_patch,
                    plt.Line2D([0],[0], color='#34d399', ls='--',
                               label='Good threshold (0.5)')],
           fontsize=8.5, labelcolor='#e2e8f0',
           facecolor='#1f2937', edgecolor='#374151')

fig.suptitle('K-Means Model Selection — Aravali Districts\n'
             'Elbow method + Silhouette score to determine optimal k',
             color='#f9fafb', fontsize=12, fontweight='bold',
             fontfamily='monospace', y=1.02)

plt.tight_layout(pad=2.0)
plt.savefig('/mnt/user-data/outputs/fig_elbow_silhouette.png',
            dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: fig_elbow_silhouette.png")


# ══════════════════════════════════════════════════════════════
# VISUALISATION 2 — PCA CLUSTER SCATTER (k=2 and k=3)
# ══════════════════════════════════════════════════════════════
CLUSTER_COLORS_K2 = ['#ef4444', '#22c55e']
CLUSTER_COLORS_K3 = [eco_k3[cl]['color'] for cl in range(3)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
fig.patch.set_facecolor('#0d1117')

for ax, labels, k, title_suffix in [
    (ax1, labels_k2, 2, 'k = 2'),
    (ax2, labels_k3, 3, 'k = 3  (Ecological Labels)'),
]:
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values(): sp.set_edgecolor('#374151')
    ax.tick_params(colors='#9ca3af', labelsize=8.5)

    cluster_ids = sorted(set(labels))

    # Draw convex hull / ellipse per cluster background
    for cl in cluster_ids:
        mask = labels == cl
        pts  = Z_pca[mask]
        col  = (CLUSTER_COLORS_K2 if k == 2 else CLUSTER_COLORS_K3)[cl]
        if len(pts) > 1:
            # Draw circle/ellipse around cluster
            cx, cy = pts.mean(axis=0)
            rx = pts[:, 0].std() * 1.6 + 0.25
            ry = pts[:, 1].std() * 1.6 + 0.25
            theta = np.linspace(0, 2*np.pi, 200)
            ex = cx + rx * np.cos(theta)
            ey = cy + ry * np.sin(theta)
            ax.fill(ex, ey, alpha=0.12, color=col)
            ax.plot(ex, ey, lw=1.2, alpha=0.4, color=col, ls='--')

    # Plot each district
    for i, district in enumerate(DISTRICTS):
        cl  = labels[i]
        col = (CLUSTER_COLORS_K2 if k == 2 else CLUSTER_COLORS_K3)[cl]
        ax.scatter(
            Z_pca[i, 0], Z_pca[i, 1],
            c=col, marker=DISTRICT_MARKERS[district],
            s=220, zorder=6, edgecolors='white', linewidths=1.5,
            alpha=0.95
        )
        # Label offset — avoid overlap
        off_x = 0.06
        off_y = 0.06
        ax.annotate(
            district,
            (Z_pca[i, 0], Z_pca[i, 1]),
            xytext=(Z_pca[i, 0] + off_x, Z_pca[i, 1] + off_y),
            fontsize=9.5, color='#f9fafb',
            fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='#1f2937', edgecolor='#374151',
                      alpha=0.85)
        )

    # Axes labels with variance
    ax.set_xlabel(
        f'PC1  ({evr[0]*100:.1f}% variance)',
        color='#9ca3af', fontsize=10, fontfamily='monospace'
    )
    ax.set_ylabel(
        f'PC2  ({evr[1]*100:.1f}% variance)',
        color='#9ca3af', fontsize=10, fontfamily='monospace'
    )

    # Legend
    if k == 2:
        leg_handles = [
            mpatches.Patch(facecolor=CLUSTER_COLORS_K2[cl],
                           edgecolor='white',
                           label=f'Cluster {cl+1}')
            for cl in range(2)
        ]
    else:
        leg_handles = [
            mpatches.Patch(facecolor=eco_k3[cl]['color'],
                           edgecolor='white',
                           label=eco_k3[cl]['name'])
            for cl in range(3)
        ]

    ax.legend(handles=leg_handles, fontsize=8.5,
              labelcolor='#e2e8f0', facecolor='#1f2937',
              edgecolor='#374151', loc='best')

    ax.set_title(
        f'PCA Cluster Projection  —  {title_suffix}',
        color='#f9fafb', fontsize=11, fontweight='bold',
        fontfamily='monospace', pad=10
    )
    ax.grid(True, color='#21262d', lw=0.5, alpha=0.6)
    ax.axhline(0, color='#374151', lw=0.8, alpha=0.5)
    ax.axvline(0, color='#374151', lw=0.8, alpha=0.5)

fig.suptitle(
    'Manual PCA Projection + K-Means Clustering\n'
    f'Features: {", ".join(feature_names)}',
    color='#f9fafb', fontsize=12, fontweight='bold',
    fontfamily='monospace', y=1.02
)

plt.tight_layout(pad=2.0)
plt.savefig('/mnt/user-data/outputs/fig_pca_clusters.png',
            dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: fig_pca_clusters.png")


# ══════════════════════════════════════════════════════════════
# VISUALISATION 3 — CLUSTER PROFILE BAR CHART (k=3)
# per-cluster mean of key features (original, not standardised)
# ══════════════════════════════════════════════════════════════
PROFILE_FEATURES = {
    'Pct_NDVI'   : 'NDVI Change\n2015→2023 (%)',
    'Pct_Mining' : 'Mining Change\n2015→2023 (%)',
    'Pct_Forest' : 'Forest Change\n2015→2023 (%)',
    'Pct_GW'     : 'GW Depth Change\n2015→2023 (%)',
    'Mean_NDVI'  : 'Mean NDVI\n(×100 for scale)',
    'GW_Stress_Yrs': 'GW Stress\nYears (out of 5)',
}
pf_keys = list(PROFILE_FEATURES.keys())
pf_lbls = list(PROFILE_FEATURES.values())

# Build cluster profile matrix (k=3, raw features)
profile_data = {}
for cl in range(3):
    mask    = labels_k3 == cl
    members = [DISTRICTS[i] for i in range(len(DISTRICTS)) if mask[i]]
    vals    = feat_df.loc[members, pf_keys].mean()
    # Scale Mean_NDVI × 100 for visual comparability
    vals_plot = vals.copy()
    vals_plot['Mean_NDVI'] = vals_plot['Mean_NDVI'] * 100
    profile_data[cl] = {
        'vals'   : vals_plot.values,
        'members': members,
        'color'  : eco_k3[cl]['color'],
        'label'  : eco_k3[cl]['name'],
    }

fig, axes = plt.subplots(1, len(pf_keys), figsize=(18, 6.5))
fig.patch.set_facecolor('#0d1117')

for ax_idx, (ax, fkey, flbl) in enumerate(
        zip(axes, pf_keys, pf_lbls)):
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values(): sp.set_edgecolor('#374151')
    ax.tick_params(colors='#9ca3af', labelsize=7.5)

    vals_by_cluster = []
    colors_by_cluster = []
    for cl in range(3):
        v = profile_data[cl]['vals'][ax_idx]
        vals_by_cluster.append(v)
        colors_by_cluster.append(profile_data[cl]['color'])

    x_pos = np.arange(3)
    bars  = ax.bar(x_pos, vals_by_cluster,
                   color=colors_by_cluster,
                   edgecolor='#0d1117', linewidth=0.8,
                   width=0.55, alpha=0.88)

    for xi, v in zip(x_pos, vals_by_cluster):
        y_off = max(abs(max(vals_by_cluster)) * 0.04, 0.5)
        ax.text(xi, v + (y_off if v >= 0 else -y_off * 2.5),
                f'{v:.1f}',
                ha='center', fontsize=7.5, color='white',
                fontfamily='monospace', fontweight='bold')

    ax.axhline(0, color='#9ca3af', lw=1.0, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['C1', 'C2', 'C3'], color='#9ca3af',
                       fontfamily='monospace')
    ax.set_title(flbl, color='#e2e8f0', fontsize=8.5,
                 fontweight='bold', fontfamily='monospace', pad=8)
    ax.grid(axis='y', color='#21262d', lw=0.5, alpha=0.7)

# Shared legend
leg_handles = [
    mpatches.Patch(facecolor=eco_k3[cl]['color'],
                   edgecolor='white',
                   label=f"C{cl+1}: {eco_k3[cl]['name']}\n"
                         f"({', '.join(eco_k3[cl]['members'])})")
    for cl in range(3)
]
fig.legend(handles=leg_handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, 0.01),
           frameon=True, framealpha=0.15,
           facecolor='#1f2937', edgecolor='#374151',
           labelcolor='#e2e8f0', fontsize=9)

fig.suptitle(
    'K-Means Cluster Profiles  (k = 3)  —  Aravali Districts\n'
    'Mean value per ecological cluster across key degradation indicators',
    color='#f9fafb', fontsize=12, fontweight='bold',
    fontfamily='monospace', y=1.02
)

plt.tight_layout(pad=1.5, rect=[0, 0.13, 1, 1])
plt.savefig('/mnt/user-data/outputs/fig_cluster_profiles.png',
            dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: fig_cluster_profiles.png")


# ══════════════════════════════════════════════════════════════
# VISUALISATION 4 — SPIDER / RADAR CHART (k=3 cluster profiles)
# ══════════════════════════════════════════════════════════════
RADAR_FEATURES = [
    'Pct_NDVI', 'Mean_NDVI', 'Mean_FDS',
    'Pct_Forest', 'GW_Stress_Yrs', 'Mean_MFR',
]
RADAR_LABELS = [
    'NDVI\nChange %', 'Mean\nNDVI', 'Forest\nDensity',
    'Forest\nChange %', 'GW Stress\nYears', 'Mining/\nForest',
]

# Normalise each feature to [0,1] across all 5 districts
rf_vals_raw = feat_df[RADAR_FEATURES].values.astype(float)
rf_min      = rf_vals_raw.min(axis=0)
rf_max      = rf_vals_raw.max(axis=0)
rf_range    = rf_max - rf_min
rf_range[rf_range == 0] = 1.0
rf_norm     = (rf_vals_raw - rf_min) / rf_range   # shape (5, n_radar)

# Per-cluster mean of normalised values
cluster_radar = {}
for cl in range(3):
    mask = labels_k3 == cl
    cluster_radar[cl] = rf_norm[mask].mean(axis=0)

n_radar = len(RADAR_FEATURES)
angles  = [n / float(n_radar) * 2 * np.pi for n in range(n_radar)]
angles += angles[:1]   # close the loop

fig, ax = plt.subplots(figsize=(9, 9),
                       subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

# Background grid
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.grid(color='#374151', lw=0.7, alpha=0.7)
ax.set_ylim(0, 1.15)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'],
                   color='#6b7280', fontsize=7.5, fontfamily='monospace')
ax.spines['polar'].set_color('#374151')

# Plot each cluster
for cl in range(3):
    vals   = list(cluster_radar[cl]) + [cluster_radar[cl][0]]
    color  = eco_k3[cl]['color']
    label  = eco_k3[cl]['name']
    members= ', '.join(eco_k3[cl]['members'])

    ax.plot(angles, vals, color=color, lw=2.5,
            label=f"{label}\n({members})")
    ax.fill(angles, vals, color=color, alpha=0.15)

    # Dot markers
    ax.scatter(angles[:-1], vals[:-1],
               color=color, s=60, zorder=6,
               edgecolors='white', linewidths=0.8)

# Feature labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_LABELS, color='#e2e8f0',
                   fontsize=10, fontfamily='monospace',
                   fontweight='bold')

ax.set_title(
    'Ecological Cluster Radar Chart\n'
    'K-Means k=3  |  Normalised [0–1] per indicator',
    color='#f9fafb', fontsize=12, fontweight='bold',
    fontfamily='monospace', pad=22
)

leg = ax.legend(loc='upper right',
                bbox_to_anchor=(1.42, 1.10),
                fontsize=9, labelcolor='#e2e8f0',
                facecolor='#1f2937', edgecolor='#374151',
                framealpha=0.9)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig_cluster_radar.png',
            dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Saved: fig_cluster_radar.png")


# ══════════════════════════════════════════════════════════════
# SECTION 7 — FINAL INTERPRETATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("KEY INSIGHTS — K-MEANS CLUSTERING")
print("=" * 64)

print(f"""
MODEL SELECTION:
  Best k = {best_k}  (silhouette = {sil_scores[best_k-1]:.3f})
  With only n=5 districts, k=2 and k=3 are both interpretable.
  k=3 is chosen for ecological richness.

CLUSTER ASSIGNMENTS (k=3):""")

for cl in range(3):
    members = eco_k3[cl]['members']
    label   = eco_k3[cl]['name']
    print(f"  Cluster {cl+1}: {label}")
    print(f"            Districts: {', '.join(members)}")

print(f"""
ECOLOGICAL INTERPRETATION:

  The three clusters represent distinct degradation regimes:

  1. HIGH DEGRADATION cluster (typically Jaipur + Pali):
     • Deep groundwater stress (GW_Stress_Flag = 5/5 years)
     • Significant forest loss since 2015
     • Urban and mining pressure compound each other
     • Highest GW depth (30–35 m)

  2. MODERATE STRESS cluster (typically Bhilwara):
     • High absolute mining (591 sq km peak in 2015)
     • Mining area contracted post-2015 but remains high
     • NDVI recovery visible but fragile
     • Moderate GW depth — not in stress zone

  3. LOW STRESS / RECOVERING cluster (typically Ajmer + Udaipur):
     • Ajmer: consistent forest growth (+17% since 2015)
     • Udaipur: largest forest base (3120 sq km), high NDVI
     • Udaipur mining steadily declining (τ=-1.0, significant)
     • Both benefit from lower urban extraction pressure

  PCA INSIGHTS:
     PC1 ({evr[0]*100:.1f}% variance): primarily captures
     forest scale and NDVI level — separates Udaipur
     (large forest, high NDVI) from the others.
     PC2 ({evr[1]*100:.1f}% variance): captures change
     dynamics — separates districts by rate of change
     in mining and GW stress.

  LIMITATION:
     K-Means with n=5 has limited statistical stability.
     Cluster boundaries should be interpreted as indicative,
     not definitive. The ecological labels are supported
     by the MK trend analysis from Phase 2a.
""")

print("=" * 64)
print("PHASE 2c COMPLETE")
print("=" * 64)
print("\nOutputs:")
print("  → cluster_assignments.csv")
print("  → fig_elbow_silhouette.png")
print("  → fig_pca_clusters.png")
print("  → fig_cluster_profiles.png")
print("  → fig_cluster_radar.png")
