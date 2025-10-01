import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
import umap
from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt

# ---------------------------------------------------
# UI CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Uncertainty Matrix – Diagnostic Landscape",
    layout="wide",
)

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def _safe_hist(x: np.ndarray, bins: int = 20) -> np.ndarray:
    """Return normalized histogram with epsilon to avoid zeros."""
    if np.all(np.isnan(x)):
        h = np.ones(bins, dtype=float)
        return h / h.sum()
    x = x[~np.isnan(x)]
    if x.size == 0:
        h = np.ones(bins, dtype=float)
        return h / h.sum()
    hist, edges = np.histogram(x, bins=bins, density=False)
    hist = hist.astype(float) + 1e-9
    hist /= hist.sum()
    return hist

def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * (scipy_entropy(p, m) + scipy_entropy(q, m))

def class_stats(df: pd.DataFrame, y: pd.Series, feature_cols):
    """Compute per-class mean, std, entropy and feature-level JS divergence."""
    classes = y.unique()
    if len(classes) != 2:
        raise ValueError("This demo expects exactly 2 classes.")
    c1, c2 = classes[0], classes[1]

    stats = {"classes": (c1, c2), "mu": {}, "sigma": {}, "H": {}, "JS": {}}

    # per-class mean and std
    for c in classes:
        df_c = df.loc[y == c, feature_cols]
        mu_c = df_c.mean(axis=0).to_dict()
        std_c = df_c.std(axis=0, ddof=0).replace(0, 1e-9).to_dict()
        for f in feature_cols:
            stats["mu"][(c, f)] = float(mu_c[f])
            stats["sigma"][(c, f)] = float(std_c[f])

    # per-class entropy and JS divergence between classes
    for f in feature_cols:
        p = _safe_hist(df.loc[y == c1, f].values)
        q = _safe_hist(df.loc[y == c2, f].values)
        H1 = -(p * np.log2(p)).sum()
        H2 = -(q * np.log2(q)).sum()
        stats["H"][(c1, f)] = float(H1)
        stats["H"][(c2, f)] = float(H2)
        stats["JS"][f] = float(jensen_shannon(p, q))

    return stats

def uncertainty_matrix(df: pd.DataFrame, y: pd.Series, feature_cols, stats):
    """
    Construct uncertainty matrix:
    For each patient-feature, value = entropy_selected / (JS + eps) * z_nearest_class
    """
    eps = 1e-9
    c1, c2 = stats["classes"]
    N = df.shape[0]
    F = len(feature_cols)
    X = np.zeros((N, F), dtype=float)

    for j, f in enumerate(feature_cols):
        mu1 = stats["mu"][(c1, f)]
        s1 = stats["sigma"][(c1, f)]
        mu2 = stats["mu"][(c2, f)]
        s2 = stats["sigma"][(c2, f)]
        H1 = stats["H"][(c1, f)]
        H2 = stats["H"][(c2, f)]
        JS = stats["JS"][f]

        v = df[f].values.astype(float)
        z1 = (v - mu1) / (s1 if s1 != 0 else eps)
        z2 = (v - mu2) / (s2 if s2 != 0 else eps)

        use_c1 = np.abs(z1) <= np.abs(z2)
        z = np.where(use_c1, z1, z2)
        H_sel = np.where(use_c1, H1, H2)

        X[:, j] = H_sel * (1.0 / (JS + eps)) * z

    return X

def fit_density_layers(embedding: np.ndarray, labels: np.ndarray, bandwidth: float = 0.6, quantile: float = 0.60):
    """
    Fit Gaussian KDE densities for each class and determine "core" regions.
    """
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("Exactly 2 classes required.")
    c1, c2 = classes

    X1 = embedding[labels == c1]
    X2 = embedding[labels == c2]

    kde1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(X1)
    kde2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(X2)

    s1 = kde1.score_samples(X1)
    s2 = kde2.score_samples(X2)
    level1 = np.quantile(s1, quantile)
    level2 = np.quantile(s2, quantile)

    def region_of(point_xy: np.ndarray):
        s1p = float(kde1.score_samples(point_xy.reshape(1, -1))[0])
        s2p = float(kde2.score_samples(point_xy.reshape(1, -1))[0])
        in1 = s1p >= level1
        in2 = s2p >= level2

        if in1 and in2:
            region = "Uncertain overlap"
        elif in1:
            region = f"{c1} core"
        elif in2:
            region = f"{c2} core"
        else:
            region = f"Between cores (leans to {c1 if s1p > s2p else c2})"
        return {"region": region}

    return kde1, kde2, level1, level2, region_of

# ---------------------------------------------------
# APP LOGIC
# ---------------------------------------------------
st.sidebar.header("Settings")

data_src = st.sidebar.radio("Data source", ["Local file (data.xlsx)", "Upload file"])
if data_src == "Upload file":
    up = st.sidebar.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
else:
    up = None

label_col_hint = st.sidebar.text_input("Label column", value="GRUP")
pos_class_name = st.sidebar.text_input("Class-1 name", value="Myocarditis")
neg_class_name = st.sidebar.text_input("Class-2 name", value="ACS")

umap_n_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 80, 30, 1)
umap_min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 0.99, 0.1, 0.01)
kde_bandwidth = st.sidebar.slider("KDE bandwidth", 0.1, 1.5, 0.6, 0.05)
core_quantile = st.sidebar.slider("Core quantile", 0.50, 0.90, 0.60, 0.01)

@st.cache_data(show_spinner=False)
def load_excel(file_or_buffer):
    return pd.read_excel(file_or_buffer)

if up is not None:
    df = load_excel(up)
else:
    df = load_excel("data.xlsx")

if label_col_hint not in df.columns:
    st.error("Label column not found.")
    st.stop()
else:
    label_col = label_col_hint

st.title("Uncertainty Matrix – Diagnostic Landscape")
st.caption(
    "This app visualizes the uncertainty-based diagnostic landscape of two clinical groups (e.g., Myocarditis vs ACS). "
    "The uncertainty matrix is constructed using class-specific entropy, JS divergence, and nearest-class z-scores."
)

# ---------------------------------------------------
# Preprocess data
# ---------------------------------------------------
numerics = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numerics if c != label_col]

# Fill missing values with column means
df_feat = df[feature_cols].copy().fillna(df[feature_cols].mean(numeric_only=True))

y_raw = df[label_col].astype(str).values
unique_labels = np.unique(y_raw)
if len(unique_labels) != 2:
    st.error("Two classes expected.")
    st.stop()

y = np.where(y_raw == unique_labels[0], pos_class_name, neg_class_name)

# ---------------------------------------------------
# Uncertainty matrix
# ---------------------------------------------------
stats = class_stats(df_feat, pd.Series(y), feature_cols)
X = uncertainty_matrix(df_feat, pd.Series(y), feature_cols, stats)
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)

# UMAP embedding
reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
Z = reducer.fit_transform(X_std)

# Logistic regression classifier (for risk score)
clf = LogisticRegression(max_iter=200).fit(X_std, y)

# KDE density layers for diagnostic landscape
kde1, kde2, level1, level2, region_of = fit_density_layers(Z, np.array(y), bandwidth=kde_bandwidth, quantile=core_quantile)

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------
left, right = st.columns([7, 5], vertical_alignment="top")

with left:
    st.subheader("Diagnostic Landscape (2D embedding)")
    fig, ax = plt.subplots(figsize=(8, 6))
    mask_pos = (np.array(y) == pos_class_name)

    # 1) Scatter points
    ax.scatter(Z[mask_pos, 0], Z[mask_pos, 1], s=16, alpha=0.75, label=pos_class_name, c="#E74C3C")
    ax.scatter(Z[~mask_pos, 0], Z[~mask_pos, 1], s=16, alpha=0.75, label=neg_class_name, c="#3498DB")

    # 2) KDE density clouds (error-cloud style)
    xx, yy = np.meshgrid(
        np.linspace(Z[:,0].min()-1, Z[:,0].max()+1, 200),
        np.linspace(Z[:,1].min()-1, Z[:,1].max()+1, 200)
    )
    grid = np.vstack([xx.ravel(), yy.ravel()]).T

    # kde1, kde2 zaten fit edilmişti
    zz1 = np.exp(kde1.score_samples(grid)).reshape(xx.shape)
    zz2 = np.exp(kde2.score_samples(grid)).reshape(xx.shape)

    ax.contourf(xx, yy, zz1, levels=20, cmap="Reds", alpha=0.2)   # Myocarditis cloud
    ax.contourf(xx, yy, zz2, levels=20, cmap="Blues", alpha=0.2)  # ACS cloud

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("New Patient Input")
    with st.form("new_patient"):
        cols = st.columns(4)
        new_vals = {}
        for i, f in enumerate(feature_cols):
            with cols[i % 4]:
                default_val = float(np.nanmean(df_feat[f].values))
                new_vals[f] = st.number_input(f, value=default_val, format="%.4f")
        submitted = st.form_submit_button("Project")

    if submitted:
        v = pd.DataFrame([new_vals])[feature_cols]
        v = v.fillna(df_feat.mean(numeric_only=True))  # Ensure no NaN
        X_new = uncertainty_matrix(v, pd.Series([pos_class_name]), feature_cols, stats)
        X_new_std = scaler.transform(X_new)
        z_new = reducer.transform(X_new_std)
        dens_info = region_of(z_new[0])
        proba = clf.predict_proba(X_new_std)[0]
        cls_order = clf.classes_.tolist()
        p_pos = float(proba[cls_order.index(pos_class_name)]) if pos_class_name in cls_order else float(proba[0])

        # plot new point
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(Z[mask_pos, 0], Z[mask_pos, 1], s=16, alpha=0.25, label=pos_class_name, c="#E74C3C")
        ax2.scatter(Z[~mask_pos, 0], Z[~mask_pos, 1], s=16, alpha=0.25, label=neg_class_name, c="#3498DB")
        ax2.scatter(z_new[0, 0], z_new[0, 1], s=180, marker="*", edgecolor="k", linewidth=1.0, label="New patient", c="#F1C40F")
        ax2.set_xlabel("UMAP-1")
        ax2.set_ylabel("UMAP-2")
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

        st.success(f"New patient projected → Region: {dens_info['region']} | {pos_class_name} probability: {p_pos:.2%}")

with right:
    st.subheader("Patient Details / Explanation")
    st.markdown(f"**Classes:** `{pos_class_name}` vs `{neg_class_name}`")
    st.markdown("**Uncertainty Matrix:** nearest-class z-score × class-specific entropy / JS divergence")
    st.markdown("- Entropy (H): information content within each class distribution")
    st.markdown("- JS divergence: separation between class distributions")
    st.markdown("- z-score: patient’s deviation relative to the nearest class mean")
    st.markdown("These combined values represent the uncertainty per feature and project the patient into a 2D diagnostic landscape with density-based confidence regions.")
