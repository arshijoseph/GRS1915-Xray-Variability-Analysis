# IMPORTING.....

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

# LOADING X RAY DATASET......

cols = ['MJD', 'intensity', 'intensity_err', 'soft_color', 'soft_color_err', 'hard_color', 'hard_color_err']
df = pd.read_csv('GRS1915-clean.color', sep=r'\s+', header=None, names=cols)

print("Shape:", df.shape)
print("MJD range:", df['MJD'].min(), "to", df['MJD'].max())
df.describe()

# Step 3: Check that the data loaded correctly and look for any missing/broken values

print("\nMissing values per column:")
print(df.isna().sum())

print("\nRows with zero or negative intensity:", (df['intensity'] <= 0).sum())

# Step 4: Split the continuous data into separate observation segments,
# based on wherever there's a large time gap between consecutive rows

time_diff = df['MJD'].diff()

gap_threshold = 10 / 1440
df['new_segment'] = (time_diff > gap_threshold) | (time_diff.isna())
df['segment_id'] = df['new_segment'].cumsum()

print("\nNumber of separate observation segments:", df['segment_id'].nunique())
print(df.groupby('segment_id').size().describe())

# Step 5: Plot X-ray intensity across the entire 15.7-year archive,
# to get our first visual look at how the black hole's brightness varies over time

step = 20
plt.figure(figsize=(14, 4))
plt.plot(df['MJD'][::step], df['intensity'][::step], '.', markersize=1, alpha=0.4)
plt.xlabel('Time (MJD)')
plt.ylabel('X-ray intensity (normalized)')
plt.title('GRS 1915+105 — full light curve (1996-2012)')
plt.tight_layout()
plt.show()

# Step 6: Zoom into ONE continuous observation segment to see the
# short-timescale structure that the full 15.7-year plot completely hides

seg_sizes = df.groupby('segment_id').size()
longest_id = seg_sizes.idxmax()
seg = df[df['segment_id'] == longest_id]

plt.figure(figsize=(14, 4))
plt.plot(seg['MJD'], seg['intensity'], '-', linewidth=0.7)
plt.xlabel('Time (MJD)')
plt.ylabel('X-ray intensity (normalized)')
plt.title(f'Single observation segment (segment {longest_id}, {len(seg)} points)')
plt.tight_layout()
plt.show()

# Step 7: Look at the overall distribution shape of intensity, soft colour,
# and hard colour across ALL 326,607 points — not over time, just their spread of values

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['intensity'], bins=100)
axes[0].set_title('Intensity')
axes[0].set_xlabel('Intensity')
axes[0].set_ylabel('Count')

axes[1].hist(df['soft_color'], bins=100, color='green')
axes[1].set_title('Soft colour')
axes[1].set_xlabel('Soft colour')

axes[2].hist(df['hard_color'], bins=100, color='red')
axes[2].set_title('Hard colour')
axes[2].set_xlabel('Hard colour')

plt.tight_layout()
plt.show()

# Step 8: Plot soft colour against hard colour together (not against time),
# to reveal the different spectral "states" the black hole moves through —
# this is the classic diagnostic plot used throughout the X-ray binary literature

plt.figure(figsize=(7, 6))
plt.hexbin(df['soft_color'], df['hard_color'], gridsize=150, cmap='inferno', bins='log', mincnt=1)
plt.xlabel('Soft colour')
plt.ylabel('Hard colour')
plt.title('GRS 1915+105 — colour-colour diagram')
plt.colorbar(label='log10(count)')
plt.tight_layout()
plt.show()

# Step 9: Measure how strongly intensity, soft colour, and hard colour
# move together, using a single correlation number for each pair

corr = df[['intensity', 'soft_color', 'hard_color']].corr()
print("\nCorrelation matrix:")
print(corr)

# Step 10: Load the radio flux dataset and confirm it looks structurally sound

radio_cols = ['MJD', 'flux_mJy']
radio_df = pd.read_csv('ryle-1995-2006.txt', sep=r'\s+', header=None, names=radio_cols)

print("Radio data shape:", radio_df.shape)
print("Radio MJD range:", radio_df['MJD'].min(), "to", radio_df['MJD'].max())
print("\nMissing values per column:")
print(radio_df.isna().sum())
radio_df.describe()
print(radio_df.describe())

# Step 11: Plot radio flux over time, same treatment as the X-ray light curve,
# to see how the jet's radio brightness varies across the observation period

plt.figure(figsize=(14, 4))
plt.plot(radio_df['MJD'], radio_df['flux_mJy'], '.', markersize=1, alpha=0.3)
plt.xlabel('Time (MJD)')
plt.ylabel('Radio flux (mJy)')
plt.title('GRS 1915+105 — radio light curve (Ryle Telescope)')
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.show()

# Step 12: Check exactly how much of the X-ray archive has radio coverage
# happening at the same time, before we attempt any merging later

xray_start, xray_end = df['MJD'].min(), df['MJD'].max()
radio_start, radio_end = radio_df['MJD'].min(), radio_df['MJD'].max()

print("X-ray coverage:", xray_start, "to", xray_end)
print("Radio coverage:", radio_start, "to", radio_end)

overlap_start = max(xray_start, radio_start)
overlap_end = min(xray_end, radio_end)
print("\nOverlapping time range:", overlap_start, "to", overlap_end)

xray_in_overlap = df[(df['MJD'] >= overlap_start) & (df['MJD'] <= overlap_end)]
pct_xray_covered = 100 * len(xray_in_overlap) / len(df)

print(f"\nX-ray rows within the overlap window: {len(xray_in_overlap)} out of {len(df)} ({pct_xray_covered:.1f}%)")


# Step 13: Test the Lomb-Scargle method on ONE segment, to search for a
# repeating (periodic) signal in the X-ray intensity — this is the core
# technique for detecting the rho class's ~50-60 second flaring

from astropy.timeseries import LombScargle

# Reuse the longest segment we already looked at visually in Step 6
seg_sizes = df.groupby('segment_id').size()
longest_id = seg_sizes.idxmax()
seg = df[df['segment_id'] == longest_id]

t = seg['MJD'].values * 86400      # convert time from days to seconds
y = seg['intensity'].values
dy = seg['intensity_err'].values

ls = LombScargle(t, y, dy)
freq, power = ls.autopower(minimum_frequency=1/300, maximum_frequency=1/20, samples_per_peak=10)

best_period = 1 / freq[np.argmax(power)]
print("\nBest-fit period for this segment (seconds):", best_period)

plt.figure(figsize=(10, 4))
plt.plot(1/freq, power)
plt.xlabel('Period (seconds)')
plt.ylabel('Lomb-Scargle power')
plt.title(f'Periodogram — segment {longest_id}')
plt.tight_layout()
plt.show()


# Step 14: Repeat the periodogram test on EVERY observation segment,
# not just one — this is how we search broadly for genuine periodic signals,
# including rho-class candidates, across the entire archive

results = []

for seg_id, seg in df.groupby('segment_id'):
    if len(seg) < 64:
        continue

    t = seg['MJD'].values * 86400
    y = seg['intensity'].values
    dy = seg['intensity_err'].values

    if np.ptp(t) < 600:
        continue

    ls = LombScargle(t, y, dy)
    freq, power = ls.autopower(minimum_frequency=1/300, maximum_frequency=1/20, samples_per_peak=10)

    best_idx = np.argmax(power)
    best_period = 1 / freq[best_idx]
    best_power = power[best_idx]

    fap = ls.false_alarm_probability(best_power, method='baluev',
                                      minimum_frequency=1/300, maximum_frequency=1/20)

    results.append({
        'segment_id': seg_id,
        'n_points': len(seg),
        'best_period_s': best_period,
        'best_power': best_power,
        'fap': fap,
        'mean_intensity': y.mean(),
        'mean_soft_color': seg['soft_color'].mean(),
        'mean_hard_color': seg['hard_color'].mean(),
    })

periodicity_df = pd.DataFrame(results)
print("\nDone! Segments analysed:", len(periodicity_df))


# Check whether any segments produced invalid (NaN) results due to the zero-error issue
print("Rows with NaN in periodicity_df:")
print(periodicity_df.isna().sum())

print("\nHow many segments contain at least one row with zero intensity_err:")
zero_err_segments = df[df['intensity_err'] == 0]['segment_id'].nunique()
print(zero_err_segments)

print("Rows with NaN in periodicity_df:")
print(periodicity_df.isna().sum())

# Quick cleanup: remove the small number of segments where the periodicity
# calculation itself failed (NaN results), before doing any further analysis

print("Rows before removing NaNs:", len(periodicity_df))
periodicity_df = periodicity_df.dropna(subset=['best_power', 'fap']).reset_index(drop=True)
print("Rows after removing NaNs:", len(periodicity_df))

print("////// 15th step//////////")

# Step 15: Decide which segments show a statistically trustworthy periodic
# signal, and specifically which ones match the rho class's known 60-120s range
# (Belloni et al. 2000, Sec 3.10: "repeats on a time scale between one and two minutes")

SIG_THRESHOLD = 0.01
RHO_MIN, RHO_MAX = 60, 120

periodicity_df['significant'] = periodicity_df['fap'] < SIG_THRESHOLD
periodicity_df['rho_candidate'] = (
    periodicity_df['significant'] &
    periodicity_df['best_period_s'].between(RHO_MIN, RHO_MAX)
)

n_sig = periodicity_df['significant'].sum()
n_rho = periodicity_df['rho_candidate'].sum()

print(f"\nSignificant periodicity (FAP < {SIG_THRESHOLD}): {n_sig} out of {len(periodicity_df)} segments")
print(f"Rho-class candidates (period 60-120s AND significant): {n_rho} segments")

# FAP threshold sensitivity test
fap_thresholds = [0.05, 0.01, 0.001, 0.0001]

print("\nFAP threshold sensitivity test:")
for thresh in fap_thresholds:
    sig = periodicity_df['fap'] < thresh
    rho = sig & periodicity_df['best_period_s'].between(RHO_MIN, RHO_MAX)
    print(f"FAP < {thresh}: {sig.sum()} significant, {rho.sum()} rho-class candidates")

# Period distribution of all significant segments
sig_df = periodicity_df[periodicity_df['significant']]

plt.figure(figsize=(10, 5))
plt.hist(sig_df['best_period_s'], bins=50, edgecolor='black')
plt.axvspan(RHO_MIN, RHO_MAX, color='orange', alpha=0.3, label='Rho-class window (60-120s)')
plt.xlabel('Best-fit period (s)')
plt.ylabel('Count')
plt.title('Period distribution of all significant segments (FAP < 0.01)')
plt.legend()
plt.tight_layout()
plt.show()

print(sig_df['best_period_s'].describe())

# Period vs. intensity, soft colour, hard colour
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(sig_df['best_period_s'], sig_df['mean_intensity'], alpha=0.5)
axes[0].set_xlabel('Period (s)'); axes[0].set_ylabel('Mean intensity')

axes[1].scatter(sig_df['best_period_s'], sig_df['mean_soft_color'], alpha=0.5, color='green')
axes[1].set_xlabel('Period (s)'); axes[1].set_ylabel('Soft colour')

axes[2].scatter(sig_df['best_period_s'], sig_df['mean_hard_color'], alpha=0.5, color='red')
axes[2].set_xlabel('Period (s)'); axes[2].set_ylabel('Hard colour')

plt.tight_layout()
plt.show()

print(sig_df[['best_period_s', 'mean_intensity', 'mean_soft_color', 'mean_hard_color']].corr())


# Step 16: Take the strongest rho-class candidate and "fold" its light curve
# on its detected period, to visually check if it forms a clean repeating shape
# (using the corrected 60-120s Belloni-range window)

rho_candidates = periodicity_df[periodicity_df['rho_candidate']]
best_row = rho_candidates.sort_values('best_power', ascending=False).iloc[0]

seg = df[df['segment_id'] == best_row['segment_id']]
t = seg['MJD'].values * 86400
t = t - t[0]
y = seg['intensity'].values

period = best_row['best_period_s']
phase = (t % period) / period

plt.figure(figsize=(10, 4))
plt.scatter(phase, y, s=15)
plt.scatter(phase + 1, y, s=15, color='C0')
plt.xlabel('Phase (0 to 2, repeated for clarity)')
plt.ylabel('X-ray intensity')
plt.title(f"Phase-folded light curve — segment {int(best_row['segment_id'])}, P={period:.1f}s")
plt.tight_layout()
plt.show()

print(f"Segment {int(best_row['segment_id'])}: period={period:.2f}s, power={best_row['best_power']:.3f}, FAP={best_row['fap']:.2e}")


# Step 16a: Check period-window sensitivity — does rho-like density
# extend meaningfully beyond Belloni's stated 60-120s range?

windows_to_test = [(50, 130), (55, 125), (60, 120), (65, 115), (70, 110)]

print("\nPeriod window sensitivity test:")
for lo, hi in windows_to_test:
    rho = periodicity_df['significant'] & periodicity_df['best_period_s'].between(lo, hi)
    print(f"Window {lo}-{hi}s: {rho.sum()} rho-class candidates")

just_below = periodicity_df[
    periodicity_df['significant'] &
    periodicity_df['best_period_s'].between(50, 60)
]
just_above = periodicity_df[
    periodicity_df['significant'] &
    periodicity_df['best_period_s'].between(120, 140)
]
print(f"\nJust below window (50-60s): {len(just_below)} significant segments")
print(f"Just above window (120-140s): {len(just_above)} significant segments")


# Step 16b: Compare three period groups against Belloni's window

group_A = periodicity_df[periodicity_df['significant'] & periodicity_df['best_period_s'].between(60, 120)]
group_B = periodicity_df[periodicity_df['significant'] & periodicity_df['best_period_s'].between(120, 140)]
group_C = periodicity_df[periodicity_df['significant'] & (periodicity_df['best_period_s'] > 140)]

print(f"\nGroup A (60-120s, Belloni range): {len(group_A)} segments")
print(f"Group B (120-140s, just outside): {len(group_B)} segments")
print(f"Group C (>140s, longer-period): {len(group_C)} segments")

print("\nGroup A — property summary:")
print(group_A[['best_period_s', 'mean_intensity', 'mean_soft_color', 'mean_hard_color', 'best_power']].describe())

print("\nGroup B — property summary:")
print(group_B[['best_period_s', 'mean_intensity', 'mean_soft_color', 'mean_hard_color', 'best_power']].describe())

print("\nGroup C — property summary:")
print(group_C[['best_period_s', 'mean_intensity', 'mean_soft_color', 'mean_hard_color', 'best_power']].describe())

for label, group in [('A (60-120s)', group_A), ('B (120-140s)', group_B), ('C (>140s)', group_C)]:
    if len(group) == 0:
        continue
    row = group.sort_values('best_power', ascending=False).iloc[0]
    seg = df[df['segment_id'] == row['segment_id']]
    t = seg['MJD'].values * 86400
    t = t - t[0]
    y = seg['intensity'].values
    period = row['best_period_s']
    phase = (t % period) / period

    plt.figure(figsize=(8, 3))
    plt.scatter(phase, y, s=15)
    plt.scatter(phase + 1, y, s=15, color='C0')
    plt.xlabel('Phase (0 to 2)')
    plt.ylabel('X-ray intensity')
    plt.title(f"Group {label} — Segment {int(row['segment_id'])}, P={period:.1f}s, power={row['best_power']:.3f}")
    plt.tight_layout()
    plt.show()


# Step 16c: Quantify light-curve shape for all 56 Group A candidates
from scipy.stats import skew

def compute_skewness(seg_id, period):
    seg = df[df['segment_id'] == seg_id]
    y = seg['intensity'].values
    return skew(y)

group_A = group_A.copy()
group_A['skewness'] = group_A.apply(
    lambda row: compute_skewness(row['segment_id'], row['best_period_s']), axis=1
)

group_A_sorted = group_A.sort_values('skewness', ascending=False)
print("\nGroup A candidates ranked by skewness (high = sharp flare-like, low/negative = rounded or dip-like):")
print(group_A_sorted[['segment_id', 'best_period_s', 'best_power', 'skewness']].to_string(index=False))

print("\nSkewness summary:")
print(group_A['skewness'].describe())

flare_like = (group_A['skewness'] > 1.0).sum()
rounded = (group_A['skewness'].between(-0.5, 1.0)).sum()
dip_like = (group_A['skewness'])
        
# Step 17: Check whether segments with significant periodicity — especially
# rho-class candidates — cluster in a specific region of colour-colour space,
# rather than being scattered randomly across it

plt.figure(figsize=(7, 6))
plt.hexbin(df['soft_color'], df['hard_color'], gridsize=150, cmap='Greys', bins='log', mincnt=1)

sig = periodicity_df[periodicity_df['significant']]
rho = periodicity_df[periodicity_df['rho_candidate']]

plt.scatter(sig['mean_soft_color'], sig['mean_hard_color'], s=15, color='blue', alpha=0.6, label='Significant periodicity')
plt.scatter(rho['mean_soft_color'], rho['mean_hard_color'], s=50, color='red', edgecolor='black', label='Rho-class candidate')

plt.xlabel('Soft colour')
plt.ylabel('Hard colour')
plt.title('Where periodic segments sit in colour-colour space')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Rho candidates plotted: {len(rho)}")

# Step 18: Compute additional variability measures for every segment —
# not just the periodicity results, but statistics describing how much
# each segment's brightness fluctuates and over what range

segment_stats = df.groupby('segment_id').agg(
    std_intensity=('intensity', 'std'),
    max_intensity=('intensity', 'max'),
    min_intensity=('intensity', 'min'),
).reset_index()

segment_stats['intensity_range'] = segment_stats['max_intensity'] - segment_stats['min_intensity']

print(segment_stats.head())
print("\nShape:", segment_stats.shape)

# Step 19: Combine periodicity results and variability statistics into
# one feature table, one row per segment

features = periodicity_df.merge(segment_stats, on='segment_id', how='left')
features['fractional_rms'] = features['std_intensity'] / features['mean_intensity']

print("Feature table shape:", features.shape)
print(features.head())

print("\nAny infinite or extremely large fractional_rms values?")
print(features['fractional_rms'].describe())
print("\nMax fractional_rms:", features['fractional_rms'].max())

# Step 19a: Correlate period with fractional RMS and intensity range

sig_features = features[features['significant']]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(sig_features['best_period_s'], sig_features['fractional_rms'], alpha=0.5, color='purple')
axes[0].set_xlabel('Period (s)'); axes[0].set_ylabel('Fractional RMS')

axes[1].scatter(sig_features['best_period_s'], sig_features['intensity_range'], alpha=0.5, color='orange')
axes[1].set_xlabel('Period (s)'); axes[1].set_ylabel('Intensity range')

plt.tight_layout()
plt.show()

print(sig_features[['best_period_s', 'fractional_rms', 'intensity_range']].corr())

# Step 20: For each X-ray segment, find the nearest-in-time radio flux
# measurement, and attach it to the feature table

segment_times = df.groupby('segment_id')['MJD'].mean().reset_index()
segment_times.columns = ['segment_id', 'segment_mjd']

features = features.merge(segment_times, on='segment_id', how='left')

features = features.sort_values('segment_mjd')
radio_sorted = radio_df.sort_values('MJD')

features = pd.merge_asof(
    features, radio_sorted,
    left_on='segment_mjd', right_on='MJD',
    direction='nearest',
    tolerance=1.0
)

print("Feature table shape after radio merge:", features.shape)
print("Segments with a matched radio value:", features['flux_mJy'].notna().sum())
print("Segments without radio coverage:", features['flux_mJy'].isna().sum())

# Step 21: Put all features on the same scale, so no single feature
# unfairly dominates the clustering just because of its raw numeric size

from sklearn.preprocessing import StandardScaler

feature_cols = ['mean_intensity', 'fractional_rms', 'intensity_range',
                'mean_soft_color', 'mean_hard_color',
                'best_period_s', 'best_power']

X = features[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape of scaled feature matrix:", X_scaled.shape)
print("\nMean of each column (should be ~0):", X_scaled.mean(axis=0).round(3))
print("Std of each column (should be ~1):", X_scaled.std(axis=0).round(3))

print("\nMissing values in each feature column:")
print(features[feature_cols].isna().sum())

# Step 22: Group segments into clusters based on similarity across all 7
# features, using K-Means, and compare against Belloni's 12 known classes

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K = 12
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
features['kmeans_label'] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, features['kmeans_label'])
print(f"\nK-Means with K={K} clusters")
print(f"Silhouette score: {sil_score:.3f}")
print("\nCluster sizes:")
print(features['kmeans_label'].value_counts().sort_index())

# Step 23: Cluster using a Gaussian Mixture Model — a different algorithm
# from K-Means, to see if it finds similar or different structure

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=K, random_state=42, n_init=10)
features['gmm_label'] = gmm.fit_predict(X_scaled)

sil_score_gmm = silhouette_score(X_scaled, features['gmm_label'])
print(f"\nGMM with {K} components")
print(f"Silhouette score: {sil_score_gmm:.3f}")
print("\nCluster sizes:")
print(features['gmm_label'].value_counts().sort_index())

# Step 24: Cluster using DBSCAN — a fundamentally different approach that
# doesn't require specifying the number of clusters in advance, and can
# identify "noise" points that don't belong to any clear cluster

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.8, min_samples=10)
features['dbscan_label'] = dbscan.fit_predict(X_scaled)

n_clusters_found = len(set(features['dbscan_label'])) - (1 if -1 in features['dbscan_label'].values else 0)
n_noise = (features['dbscan_label'] == -1).sum()

print(f"\nDBSCAN found {n_clusters_found} clusters")
print(f"Noise points (unclassified): {n_noise} out of {len(features)}")

non_noise_mask = features['dbscan_label'] != -1
sil_score_dbscan = silhouette_score(X_scaled[non_noise_mask], features['dbscan_label'][non_noise_mask])
print(f"Silhouette score (excluding noise): {sil_score_dbscan:.3f}")

print("\nCluster sizes:")
print(features['dbscan_label'].value_counts().sort_index())

# Step 25: Compare all three clustering algorithms side-by-side using
# two different quality metrics, to draw an honest overall conclusion

from sklearn.metrics import davies_bouldin_score

db_kmeans = davies_bouldin_score(X_scaled, features['kmeans_label'])
db_gmm = davies_bouldin_score(X_scaled, features['gmm_label'])
db_dbscan = davies_bouldin_score(X_scaled[non_noise_mask], features['dbscan_label'][non_noise_mask])

comparison = pd.DataFrame({
    'Algorithm': ['K-Means', 'GMM', 'DBSCAN'],
    'N_clusters': [K, K, n_clusters_found],
    'Silhouette_score': [sil_score, sil_score_gmm, sil_score_dbscan],
    'Davies_Bouldin_score': [db_kmeans, db_gmm, db_dbscan],
})

print(comparison)
comparison.to_csv('clustering_comparison.csv', index=False)

# Step 26: Build the final catalog — one row per segment, with its time,
# assigned class (from K-Means, our best-performing algorithm), and
# all the features that describe it

catalog = features[[
    'segment_id', 'segment_mjd', 'kmeans_label',
    'mean_intensity', 'fractional_rms', 'intensity_range',
    'mean_soft_color', 'mean_hard_color',
    'best_period_s', 'best_power', 'fap', 'rho_candidate',
    'flux_mJy'
]].copy()

catalog = catalog.rename(columns={'kmeans_label': 'variability_class'})
catalog = catalog.sort_values('segment_mjd').reset_index(drop=True)

print("Catalog shape:", catalog.shape)
print(catalog.head(10))

catalog.to_csv('GRS1915_variability_catalog_corrected.csv', index=False)
print("\nSaved to: GRS1915_variability_catalog_corrected.csv")


# Step 27: Characterize each K-Means cluster by its typical properties,
# to compare against Belloni's published class descriptions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

cluster_profile = catalog.groupby('variability_class').agg(
    n_segments=('segment_id', 'count'),
    mean_intensity=('mean_intensity', 'mean'),
    mean_fractional_rms=('fractional_rms', 'mean'),
    mean_soft_color=('mean_soft_color', 'mean'),
    mean_hard_color=('mean_hard_color', 'mean'),
    pct_significant_period=('fap', lambda x: (x < 0.01).mean() * 100),
    n_rho_candidates=('rho_candidate', 'sum'),
    mean_radio_flux=('flux_mJy', 'mean'),
).round(3)

print(cluster_profile)
cluster_profile.to_csv('cluster_profiles_corrected.csv')
print("\nSaved to: cluster_profiles_corrected.csv")

# Step 27a — Compare clusters 1 and 6 directly, and check all pairwise distances

clusters_1_6 = cluster_profile.loc[[1, 6]]
print("Clusters 1 and 6 — side by side comparison:")
print(clusters_1_6.T)

centroid_1 = X_scaled[features['kmeans_label'] == 1].mean(axis=0)
centroid_6 = X_scaled[features['kmeans_label'] == 6].mean(axis=0)
dist_1_6 = np.linalg.norm(centroid_1 - centroid_6)
print(f"\nDistance between cluster 1 and cluster 6 centroids: {dist_1_6:.3f}")

from scipy.spatial.distance import pdist, squareform
all_centroids = np.array([X_scaled[features['kmeans_label'] == k].mean(axis=0) for k in range(K)])
dist_matrix = squareform(pdist(all_centroids))
print("\nAll pairwise centroid distances (rows/cols = cluster 0-11):")
print(pd.DataFrame(dist_matrix).round(2))
print(f"\nMean pairwise distance across all cluster pairs: {dist_matrix[np.triu_indices(K, k=1)].mean():.3f}")


# Step 27b — Look at all 56 rho candidates individually, sorted by cluster

rho_ids = catalog[catalog['rho_candidate']]['segment_id'].tolist()
rho_detail = features[features['segment_id'].isin(rho_ids)][
    ['segment_id', 'kmeans_label'] + feature_cols
].sort_values('kmeans_label')

print(f"\nAll {len(rho_detail)} rho candidates — feature values and assigned cluster:")
print(rho_detail.to_string(index=False))

print("\nRho candidate count by cluster:")
print(rho_detail['kmeans_label'].value_counts().sort_index())

# Zoom in specifically on the 8 "stray" candidates outside clusters 1 and 6
stray = rho_detail[~rho_detail['kmeans_label'].isin([1, 6])]
print(f"\nStray candidates (not in clusters 1 or 6): {len(stray)}")
print(stray.to_string(index=False))

# Step 28a: Colour-colour diagram, with each point coloured by its
# assigned K-Means cluster — the visual counterpart to the cluster_profile table

plt.figure(figsize=(8, 7))
scatter = plt.scatter(
    catalog['mean_soft_color'], catalog['mean_hard_color'],
    c=catalog['variability_class'], cmap='tab20', s=15, alpha=0.7
)
plt.xlabel('Soft colour')
plt.ylabel('Hard colour')
plt.title('GRS 1915+105 — segments coloured by discovered class')
plt.colorbar(scatter, label='Variability class (K-Means)', ticks=range(12))
plt.tight_layout()
plt.show()

# Step 28b: Class occurrence over time — does each class appear throughout
# the whole 15.7-year archive, or concentrate in specific eras?

plt.figure(figsize=(14, 6))
for cls in sorted(catalog['variability_class'].unique()):
    subset = catalog[catalog['variability_class'] == cls]
    plt.scatter(subset['segment_mjd'], [cls] * len(subset), s=8, alpha=0.6)

plt.xlabel('Time (MJD)')
plt.ylabel('Variability class')
plt.title('GRS 1915+105 — class occurrence over the full archive')
plt.yticks(range(12))
plt.tight_layout()
plt.show()
