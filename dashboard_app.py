"""
Dashboard Prediksi & Evaluasi Kinerja Trucking Multi-Vendor
PT Leschaco Logistic Indonesia (versi eksperimen)

Cara menjalankan:
    1. Taruh file-file berikut dalam satu folder yang sama dengan file ini:
       - model_random_forest.joblib
       - model_naive_bayes.joblib
       - label_encoders.joblib
       - vendor_ontime_rate.joblib
       - nb_bin_edges.joblib
       - route_jarak_lookup.joblib
       - model_metrics.json
       - hasil_prediksi_test.csv
       - ringkasan_performa_vendor.csv
    2. Install dependency: pip install -r requirements.txt
    3. Jalankan: streamlit run dashboard_app.py
    4. (Opsional, untuk publish online) deploy gratis via https://streamlit.io/cloud
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Kinerja Trucking - Leschaco",
    page_icon="🚚",
    layout="wide",
)

WINDOW_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Medium-High"}
WINDOW_RANGE = {0: "06:00-06:59", 1: "07:00-07:59", 2: "08:00-08:59", 3: "09:00-09:59"}
WINDOW_COLOR = {"Low": "#2ca02c", "Medium": "#ffbb33", "High": "#d62728", "Medium-High": "#ff8800"}


# ----------------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("model_random_forest.joblib")
    nb_model = joblib.load("model_naive_bayes.joblib")
    encoders = joblib.load("label_encoders.joblib")
    vendor_ontime_rate = joblib.load("vendor_ontime_rate.joblib")
    bin_edges = joblib.load("nb_bin_edges.joblib")
    with open("model_metrics.json") as f:
        metrics = json.load(f)
    route_jarak_lookup = {}
    if os.path.exists("route_jarak_lookup.joblib"):
        route_jarak_lookup = joblib.load("route_jarak_lookup.joblib")
    return rf_model, nb_model, encoders, vendor_ontime_rate, bin_edges, metrics, route_jarak_lookup


@st.cache_data
def load_data():
    hasil = pd.read_csv("hasil_prediksi_test.csv")
    ringkasan = pd.read_csv("ringkasan_performa_vendor.csv")
    return hasil, ringkasan


def estimasi_leadtime_dari_jarak(km):
    if km <= 20: return 4
    if km <= 50: return 6
    if km <= 100: return 8
    if km <= 150: return 10
    if km <= 250: return 14
    return 18


try:
    rf_model, nb_model, encoders, vendor_ontime_rate, bin_edges, metrics, route_jarak_lookup = load_artifacts()
    hasil_prediksi, ringkasan_vendor = load_data()
    ARTIFACTS_OK = True
except FileNotFoundError as e:
    ARTIFACTS_OK = False
    MISSING_FILE = str(e)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🚚 Leschaco Trucking Dashboard")
st.sidebar.caption("Versi eksperimen — data disimulasikan berdasar pola historis, bukan data resmi perusahaan.")
page = st.sidebar.radio(
    "Menu",
    ["📊 Ringkasan & Ground Truth", "🏆 Evaluasi Performa Vendor", "🚦 Analisis Traffic Risk",
     "🔮 Prediksi Pengiriman Baru", "🧪 Perbandingan Model"],
)

if not ARTIFACTS_OK:
    st.error(
        f"File pendukung belum ditemukan di folder ini ({MISSING_FILE}).\n\n"
        "Pastikan semua file .joblib, .json, dan .csv hasil notebook Colab sudah "
        "diletakkan satu folder dengan dashboard_app.py ini."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Halaman 1: Ringkasan & Ground Truth
# ----------------------------------------------------------------------------
if page == "📊 Ringkasan & Ground Truth":
    st.title("📊 Ringkasan Kinerja Trucking Multi-Vendor")
    st.caption("`on_time` di sini adalah ground truth: dibentuk dari delay aktual (actual leadtime - estimasi) "
               "dibandingkan toleransi — bukan prediksi model.")

    col1, col2, col3, col4 = st.columns(4)
    total_kirim = len(hasil_prediksi)
    ontime_rate = hasil_prediksi["actual_on_time"].mean()
    n_vendor = ringkasan_vendor["vendor_clean"].nunique()
    rf_acc = metrics["random_forest"]["accuracy"]

    col1.metric("Total Pengiriman (data uji)", f"{total_kirim:,}")
    col2.metric("On-Time Rate (Ground Truth)", f"{ontime_rate:.1%}")
    col3.metric("Jumlah Vendor Aktif", f"{n_vendor}")
    col4.metric("Akurasi Model Terbaik", f"{max(rf_acc, metrics['naive_bayes']['accuracy']):.1%}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi Ground Truth: On Time vs Not On Time")
        pie_df = hasil_prediksi["actual_on_time"].map({1: "Tepat Waktu", 0: "Tidak Tepat Waktu"}).value_counts().reset_index()
        pie_df.columns = ["Status", "Jumlah"]
        fig = px.pie(pie_df, names="Status", values="Jumlah", hole=0.45,
                     color="Status",
                     color_discrete_map={"Tepat Waktu": "#2ca02c", "Tidak Tepat Waktu": "#d62728"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("On-Time Rate per Bulan Booking")
        bulan_map_id = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
                         7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}
        trend = hasil_prediksi.groupby("booking_month")["actual_on_time"].mean().reset_index()
        trend["Bulan"] = trend["booking_month"].map(bulan_map_id)
        fig2 = px.bar(trend, x="Bulan", y="actual_on_time", text_auto=".0%",
                      labels={"actual_on_time": "On-Time Rate"})
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("On-Time Rate per Kota Tujuan (Top 15)")
    top_kota = (hasil_prediksi.groupby("kota_tujuan")["actual_on_time"]
                .agg(["mean", "count"]).query("count >= 15")
                .sort_values("count", ascending=False).head(15).reset_index())
    fig3 = px.bar(top_kota, x="kota_tujuan", y="mean", text_auto=".0%",
                  labels={"mean": "On-Time Rate", "kota_tujuan": "Kota Tujuan"})
    fig3.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# Halaman 2: Evaluasi Performa Vendor
# ----------------------------------------------------------------------------
elif page == "🏆 Evaluasi Performa Vendor":
    st.title("🏆 Evaluasi Performa Vendor (berdasarkan Ground Truth Aktual)")
    st.caption("Ranking di halaman ini memakai hasil aktual (ground truth), bukan prediksi model — "
               "supaya benar-benar mencerminkan kinerja vendor sesungguhnya.")

    min_kirim = st.slider("Minimal jumlah pengiriman untuk ditampilkan", 1, 200, 30)
    view = ringkasan_vendor[ringkasan_vendor["total_pengiriman"] >= min_kirim].copy()
    view = view.sort_values("ontime_rate_aktual", ascending=False)

    st.subheader("Peringkat Vendor berdasarkan On-Time Rate Aktual")
    fig = px.bar(view, x="vendor_clean", y="ontime_rate_aktual", text_auto=".0%",
                 color="ontime_rate_aktual", color_continuous_scale="RdYlGn",
                 labels={"ontime_rate_aktual": "On-Time Rate", "vendor_clean": "Vendor"})
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rata-rata Delay per Vendor (jam)")
        fig2 = px.bar(view.sort_values("avg_delay_jam"), x="avg_delay_jam", y="vendor_clean",
                      orientation="h", color="avg_delay_jam", color_continuous_scale="Reds",
                      labels={"avg_delay_jam": "Rata-rata Delay (jam)", "vendor_clean": "Vendor"})
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Akar Masalah Keterlambatan per Vendor")
        melt = view[["vendor_clean", "pct_faktor_traffic", "pct_faktor_loading"]].melt(
            id_vars="vendor_clean", var_name="faktor", value_name="proporsi")
        melt["faktor"] = melt["faktor"].map({"pct_faktor_traffic": "Traffic", "pct_faktor_loading": "Loading"})
        fig3 = px.bar(melt, x="vendor_clean", y="proporsi", color="faktor", barmode="stack",
                      text_auto=".0%", color_discrete_map={"Traffic": "#d62728", "Loading": "#ff8800"})
        fig3.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Tabel Lengkap Performa Vendor")
    st.dataframe(
        view.style.format({
            "ontime_rate_aktual": "{:.1%}", "avg_delay_jam": "{:.2f} jam",
            "pct_faktor_traffic": "{:.1%}", "pct_faktor_loading": "{:.1%}",
        }),
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# Halaman 3: Analisis Traffic Risk
# ----------------------------------------------------------------------------
elif page == "🚦 Analisis Traffic Risk":
    st.title("🚦 Kontribusi Traffic Risk terhadap Keterlambatan")

    kontribusi = metrics.get("kontribusi_traffic_risk")
    if kontribusi:
        st.metric("Kontribusi traffic risk & jam keberangkatan pada model prediksi", f"{kontribusi:.1%}")
        st.caption("Dihitung dari total feature importance kolom `traffic_risk_code` + `recommended_departure` pada Random Forest.")

    st.divider()
    order = ["Low", "Medium", "High", "Medium-High"]
    if "traffic_risk_code" in hasil_prediksi.columns:
        hp = hasil_prediksi.copy()
        hp["traffic_risk"] = hp["traffic_risk_code"].map(WINDOW_LABELS)
        agg = hp.groupby("traffic_risk")["actual_on_time"].agg(["mean", "count"]).reindex(order)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("On-Time Rate per Window Keberangkatan")
            fig = px.bar(agg.reset_index(), x="traffic_risk", y="mean", text_auto=".0%",
                         color="traffic_risk", color_discrete_map=WINDOW_COLOR,
                         labels={"mean": "On-Time Rate", "traffic_risk": "Traffic Risk"})
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Jumlah Pengiriman per Window")
            fig2 = px.bar(agg.reset_index(), x="traffic_risk", y="count",
                          color="traffic_risk", color_discrete_map=WINDOW_COLOR,
                          labels={"count": "Jumlah Pengiriman", "traffic_risk": "Traffic Risk"})
            st.plotly_chart(fig2, use_container_width=True)

        st.info(
            "🟢 **Low (06:00-06:59)** = sesuai jam standar perusahaan, secara skema selalu On Time. "
            "🟠🔴 Semakin siang jam keberangkatan (Medium/High/Medium-High), semakin tinggi risiko Not On Time — "
            "jadi kepatuhan vendor terhadap jam berangkat yang direkomendasikan adalah faktor kunci."
        )
    else:
        st.warning("Kolom traffic_risk_code tidak ditemukan di hasil_prediksi_test.csv.")

# ----------------------------------------------------------------------------
# Halaman 4: Prediksi Pengiriman Baru
# ----------------------------------------------------------------------------
elif page == "🔮 Prediksi Pengiriman Baru":
    st.title("🔮 Prediksi Ketepatan Waktu Pengiriman")
    st.caption("Isi detail rencana pengiriman — termasuk jam keberangkatan — untuk memprediksi kemungkinan tepat waktu.")

    vendor_list = sorted(vendor_ontime_rate.index.tolist())
    loading_list = sorted(encoders["loading"].classes_.tolist())
    unloading_list = sorted(encoders["unloading"].classes_.tolist())
    kota_list = sorted(encoders["kota_tujuan"].classes_.tolist())

    with st.form("prediksi_form"):
        c1, c2 = st.columns(2)
        with c1:
            vendor = st.selectbox("Vendor", vendor_list)
            loading = st.selectbox("Lokasi Loading", loading_list)
            unloading = st.selectbox("Lokasi Unloading", unloading_list)
            kota_tujuan = st.selectbox("Kota Tujuan", kota_list)
            jam_berangkat_window = st.select_slider(
                "Rencana Jam Keberangkatan", options=[0, 1, 2, 3],
                format_func=lambda w: f"{WINDOW_LABELS[w]} ({WINDOW_RANGE[w]})",
            )
        with c2:
            container_size = st.selectbox("Ukuran Kontainer", [20, 40], index=1)
            container_qty = st.number_input("Jumlah Kontainer", min_value=1, max_value=10, value=1)
            is_dg = st.checkbox("Dangerous Goods (DG)?")
            booking_month = st.selectbox("Bulan Booking", list(range(1, 13)), index=0)
            booking_dayofweek = st.selectbox(
                "Hari Booking", list(range(7)),
                format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x],
            )
        submitted = st.form_submit_button("Prediksi Sekarang")

    if submitted:
        booking_is_weekend = int(booking_dayofweek in [5, 6])
        v_rate = vendor_ontime_rate.get(vendor, vendor_ontime_rate.mean())

        route_key = f"{loading}|{kota_tujuan}"
        jarak_km = route_jarak_lookup.get(route_key, 50)
        estimasi_leadtime_jam = estimasi_leadtime_dari_jarak(jarak_km)
        recommended_departure = 1 if jam_berangkat_window == 0 else 0

        row = pd.DataFrame([{
            "vendor_clean": vendor, "loading": loading, "unloading": unloading,
            "kota_tujuan": kota_tujuan, "jarak_km": jarak_km,
            "estimasi_leadtime_jam": estimasi_leadtime_jam,
            "container_size": container_size, "container_qty": container_qty, "is_dg": int(is_dg),
            "booking_month": booking_month, "booking_dayofweek": booking_dayofweek,
            "booking_is_weekend": booking_is_weekend,
            "traffic_risk_code": jam_berangkat_window,
            "recommended_departure": recommended_departure,
            "vendor_ontime_rate": v_rate,
        }])

        row_enc = row.copy()
        for c in ["vendor_clean", "loading", "unloading", "kota_tujuan"]:
            le = encoders[c]
            known = set(le.classes_)
            row_enc[c] = row_enc[c].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            row_enc[c] = le.transform(row_enc[c])

        proba_rf = rf_model.predict_proba(row_enc[rf_model.feature_names_in_])[0, 1]
        pred_rf = int(proba_rf >= 0.5)

        st.divider()
        st.caption(f"Rute {loading} → {kota_tujuan}: jarak diestimasi {jarak_km} km, "
                   f"estimasi leadtime {estimasi_leadtime_jam} jam, window keberangkatan: "
                   f"{WINDOW_LABELS[jam_berangkat_window]} ({WINDOW_RANGE[jam_berangkat_window]}).")
        st.subheader("Hasil Prediksi (Random Forest)")
        colA, colB = st.columns(2)
        colA.metric("Prediksi", "✅ Tepat Waktu" if pred_rf == 1 else "⚠️ Berisiko Terlambat")
        colB.metric("Probabilitas Tepat Waktu", f"{proba_rf:.1%}")

        if proba_rf < 0.5:
            st.warning(
                f"Vendor **{vendor}** untuk rute **{loading} → {unloading}** menuju **{kota_tujuan}** "
                f"berangkat di window **{WINDOW_LABELS[jam_berangkat_window]}** berisiko tidak tepat waktu. "
                "Pertimbangkan menjadwalkan keberangkatan lebih pagi (window Low, 06:00-06:59)."
            )
        else:
            st.success("Kombinasi vendor, rute, dan jam keberangkatan ini secara historis cenderung tepat waktu.")

# ----------------------------------------------------------------------------
# Halaman 5: Perbandingan Model
# ----------------------------------------------------------------------------
elif page == "🧪 Perbandingan Model":
    st.title("🧪 Perbandingan Random Forest vs Naive Bayes")
    st.caption("Akurasi dihitung dari prediksi model (fitur sebelum pengiriman selesai) dibandingkan ground truth "
               "(dibentuk dari delay aktual & toleransi) pada data uji.")

    comp_df = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
        "Random Forest": [metrics["random_forest"]["accuracy"], metrics["random_forest"]["precision"],
                           metrics["random_forest"]["recall"], metrics["random_forest"]["f1"],
                           metrics["random_forest"].get("auc", None)],
        "Naive Bayes": [metrics["naive_bayes"]["accuracy"], metrics["naive_bayes"]["precision"],
                         metrics["naive_bayes"]["recall"], metrics["naive_bayes"]["f1"],
                         metrics["naive_bayes"].get("auc", None)],
    })

    fig = px.bar(comp_df.melt(id_vars="Metrik", var_name="Model", value_name="Skor"),
                 x="Metrik", y="Skor", color="Model", barmode="group", text_auto=".2f")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comp_df.style.format({"Random Forest": "{:.1%}", "Naive Bayes": "{:.1%}"}), use_container_width=True)

    better = "Random Forest" if metrics["random_forest"]["f1"] > metrics["naive_bayes"]["f1"] else "Naive Bayes"
    st.success(f"**{better}** memberikan F1-Score lebih tinggi dalam memprediksi ketepatan waktu pada eksperimen ini.")

    st.info(
        "**Catatan metodologi:** Fitur model (vendor, rute, jarak, estimasi leadtime, jenis kontainer, waktu "
        "booking, traffic risk, jam keberangkatan) semuanya diketahui **sebelum pengiriman selesai** — bukan "
        "leakage. Target (`on_time`) adalah ground truth yang dibentuk dari delay aktual dibandingkan toleransi "
        "**setelah** pengiriman selesai, sehingga akurasi di atas benar-benar mengukur kemampuan prediksi model, "
        "bukan sekadar membaca ulang hasil yang sudah terjadi."
    )

    with st.expander("Parameter Random Forest Terbaik (hasil GridSearchCV)"):
        st.json(metrics.get("best_rf_params", {}))
