import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# KENDİ NOT DEFTERİNDEN ÜRETTİĞİN MODÜL:
import hier_uncertainty as HU  # <-- hier_uncertainty.py ile aynı klasörde olmalı

st.set_page_config(page_title="Hierarchical Uncertainty", layout="wide")
st.title("Hierarchical Uncertainty – Demo")

# ---------- Yardımcılar ----------
@st.cache_data(show_spinner=False)
def load_excel(file_or_buffer):
    return pd.read_excel(file_or_buffer)

def df_from_upload(upload):
    if upload is None:
        return None
    data = upload.read()
    return load_excel(io.BytesIO(data))

def draw_heatmap(U, title="Uncertainty Matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(U, cmap="magma", cbar=True, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# ---------- Veri seçimi ----------
left, right = st.columns([1,1])
with left:
    source = st.radio("Veri kaynağı", ["Repo’daki data.xlsx", "Dosya yükle"], index=0)
    if source == "Repo’daki data.xlsx":
        try:
            df = load_excel("data.xlsx")
            st.success("Repo’daki data.xlsx yüklendi.")
        except Exception as e:
            st.error(f"data.xlsx okunamadı: {e}")
            df = None
    else:
        up = st.file_uploader("XLSX yükle", type=["xlsx"])
        df = df_from_upload(up)

with right:
    if df is not None:
        st.caption(f"Satır/Sütun: {df.shape[0]} / {df.shape[1]}")
        st.dataframe(df.head(10), use_container_width=True)

if df is None:
    st.stop()

# ---------- Etiket sütunu + isimler ----------
cols = df.columns.tolist()
label_col = st.selectbox("Sınıf sütunu (label)", options=cols, index=cols.index("GRUP") if "GRUP" in cols else 0)
c1 = st.text_input("Class-1 adı", value="Myocarditis")
c2 = st.text_input("Class-2 adı", value="ACS")

# ---------- Parametreler (NOT: kendi koduna göre kullan) ----------
with st.expander("Parametreler", expanded=True):
    pcols = st.columns(3)
    with pcols[0]:
        n_bins = st.slider("Sürekli değişkenler için quantile bin sayısı", 3, 20, 5)
    with pcols[1]:
        core_q = st.slider("Core quantile", 0.50, 0.90, 0.60, step=0.01)
    with pcols[2]:
        bandwidth = st.slider("KDE bandwidth", 0.10, 1.50, 0.60, step=0.05)

params = dict(n_bins=n_bins, core_quantile=core_q, bandwidth=bandwidth,
              class_names=(c1, c2))

# ---------- Çalıştır ----------
run = st.button("Run pipeline")
if run:
    t0 = time.time()
    with st.spinner("Pipeline çalışıyor…"):
        try:
            U, info = HU.run_pipeline(df, label_col, params)  # ← SENİN FONKSİYONUN
        except AttributeError:
            st.error("`hier_uncertainty.py` içinde `run_pipeline(df, label_col, params)` fonksiyonunu tanımla.")
            st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()
    dt = time.time() - t0
    st.success(f"Tamamlandı ({dt:.2f} sn)")

    # 1) UM matrisi
    if isinstance(U, (pd.DataFrame, np.ndarray)):
        if isinstance(U, np.ndarray):
            U = pd.DataFrame(U)
        draw_heatmap(U, title="Uncertainty Matrix (U)")
        # indir
        csv = U.to_csv(index=False).encode("utf-8")
        st.download_button("UM matrisi (CSV) indir", csv, "uncertainty_matrix.csv", "text/csv")
    else:
        st.warning("`run_pipeline` UM matrisini pandas DataFrame veya numpy array olarak döndürmeli.")

    # 2) Özet/Metrikler
    with st.expander("Özet/Metrikler", expanded=True):
        if isinstance(info, dict):
            for k, v in info.items():
                st.write(f"**{k}:** {v}")
        else:
            st.info("Ek metrik döndürmek istersen `info` sözlüğüne ekle.")

st.caption("Not: Bu arayüz, notebook’taki fonksiyonu çağıran bir **sarmalayıcıdır**. Algoritma tamamen sizin kodunuzdur.")
