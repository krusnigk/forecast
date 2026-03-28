import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import math
import io

# ==========================================
# PENGATURAN HALAMAN STREAMLIT
# ==========================================
st.set_page_config(page_title="WFM Forecast & Headcount App", layout="wide")
st.title("📊 WFM Call Center Forecasting & Headcount Calculator")
st.markdown("Aplikasi peramalan volume panggilan (COF) dan perhitungan kebutuhan agen menggunakan Prophet dan Erlang C.")

# ==========================================
# FUNGSI MATEMATIKA ERLANG C
# ==========================================
def erlang_c(A, N):
    if A <= 0: return 0.0
    if N <= A: return 1.0
    log_sum_terms = -float('inf')
    for k in range(N):
        try:
            term_log = k * math.log(A) - math.lgamma(k + 1)
            if log_sum_terms == -float('inf'): log_sum_terms = term_log
            else: log_sum_terms = np.logaddexp(log_sum_terms, term_log)
        except ValueError: pass
    try:
        log_numerator = (N * math.log(A) - math.lgamma(N + 1)) + (math.log(N) - math.log(N - A))
    except (ValueError, OverflowError): return 1.0 
    log_denominator = np.logaddexp(log_sum_terms, log_numerator)
    if log_denominator == -float('inf'): return 1.0 
    return max(0.0, min(1.0, math.exp(log_numerator - log_denominator)))

def calculate_agents_erlang_c(daily_calls, aht_seconds, target_sl, target_asa_seconds, shrinkage_rate, hours_per_agent_day):
    if daily_calls <= 0 or hours_per_agent_day <= 0: return 0
    workload_in_erlang = (daily_calls * aht_seconds) / 3600
    A_hourly = ((daily_calls / hours_per_agent_day) * aht_seconds) / 3600 
    found_agents = math.ceil((workload_in_erlang / hours_per_agent_day) / (1 - shrinkage_rate))
    if found_agents == 0: return 0

    for i in range(200):
        N_effective = max(1, math.ceil(found_agents * (1 - shrinkage_rate)))
        if A_hourly <= 0: return math.ceil(found_agents)
        prob_delay = 1.0 if N_effective <= A_hourly else erlang_c(A_hourly, N_effective)
        if N_effective - A_hourly <= 0: current_sl = 0.0
        else: current_sl = 1 - prob_delay * math.exp(-((N_effective - A_hourly) * target_asa_seconds) / aht_seconds)
        
        if current_sl >= target_sl: return math.ceil(found_agents)
        else: found_agents += 1
    return math.ceil(found_agents)

def calculate_actual_sl(daily_forecast_value, agents_needed_for_day, aht_seconds, target_asa_seconds, hours_per_agent_day):
    if daily_forecast_value <= 0 or agents_needed_for_day <= 0: return 1.0 
    N_effective = max(1, math.ceil(agents_needed_for_day * (1 - shrinkage_rate)))
    A_hourly = ((daily_forecast_value / hours_per_agent_day) * aht_seconds) / 3600
    if A_hourly <= 0: return 1.0 
    prob_delay = 1.0 if N_effective <= A_hourly else erlang_c(A_hourly, N_effective)
    if N_effective - A_hourly <= 0: return 0.0
    return max(0.0, min(1.0, 1 - prob_delay * math.exp(-((N_effective - A_hourly) * target_asa_seconds) / aht_seconds)))

# ==========================================
# SIDEBAR: KONFIGURASI & UPLOAD DATA
# ==========================================
st.sidebar.header("📂 1. Upload Data Excel")
uploaded_data = st.sidebar.file_uploader("Upload Data Historis (COF)", type=['xlsx'])
uploaded_holidays = st.sidebar.file_uploader("Upload Data Libur (Opsional)", type=['xlsx'])

st.sidebar.header("⚙️ 2. Konfigurasi Erlang C")
aht_seconds = st.sidebar.number_input("AHT (Detik)", min_value=10, value=480, step=10)
target_sl = st.sidebar.slider("Target Service Level", 0.1, 1.0, 0.90, 0.01)
target_asa_seconds = st.sidebar.number_input("Target ASA (Detik)", min_value=1, value=20)
shrinkage_rate = st.sidebar.slider("Shrinkage Rate", 0.0, 0.5, 0.22, 0.01)
hours_per_agent_day = st.sidebar.number_input("Jam Kerja/Agen/Hari", min_value=1, value=8)
efficiency_factor = st.sidebar.slider("Faktor Efisiensi", 0.1, 1.0, 0.85, 0.01)

st.sidebar.header("📅 3. Konfigurasi Tanggal")
start_forecast = st.sidebar.date_input("Mulai Forecast", pd.to_datetime('2026-03-01'))
end_forecast = st.sidebar.date_input("Akhir Forecast", pd.to_datetime('2026-03-31'))

# ==========================================
# AREA UTAMA: TOGGLE SCRIPT 2 & SCRIPT 3
# ==========================================
st.write("### 🛠️ Pengaturan Model Prophet")
col_t1, col_t2 = st.columns(2)
with col_t1:
    # TOGGLE UNTUK SCRIPT 2
    use_anomaly_filter = st.checkbox("✅ Aktifkan Filter Anomali (Script 2)", value=True, help="Menghapus data anomali pada Maret 2025 & Feb-Mar 2026")
with col_t2:
    # TOGGLE UNTUK SCRIPT 3
    use_seasonality = st.checkbox("🌊 Aktifkan Seasonality Bulanan (Script 3)", value=True, help="Memaksa Prophet mencari pola siklus per ~30.5 hari")

# ==========================================
# PROSES UTAMA
# ==========================================
if uploaded_data is not None:
    try:
        # Membaca data
        df = pd.read_excel(uploaded_data)
        # Asumsi kolom bernama 'date' dan 'cof' berdasarkan script asli
        df = df.rename(columns={'date': 'ds', 'cof': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
        
        # Tambahan Regressor Tanggal 1
        df['tanggal_satu'] = (df['ds'].dt.day == 1).astype(int)

        df_historical = df.copy()

        # --- LOGIKA SCRIPT 2: PENANGANAN ANOMALI ---
        if use_anomaly_filter:
            anomali_1 = (df_historical['ds'] >= '2025-03-01') & (df_historical['ds'] <= '2025-03-31')
            anomali_2 = (df_historical['ds'] >= '2026-02-19') & (df_historical['ds'] <= '2026-03-22')
            kondisi_anomali = anomali_1 | anomali_2
            jumlah_anomali = kondisi_anomali.sum()
            df_historical.loc[kondisi_anomali, 'y'] = np.nan
            st.info(f"Filter Anomali Aktif: {jumlah_anomali} baris data dikosongkan (diubah jadi NaN) sebelum di-training.")

        # --- LOGIKA HARI LIBUR ---
        holidays_df = None
        if uploaded_holidays is not None:
            df_hol = pd.read_excel(uploaded_holidays)
            if 'Tanggal' in df_hol.columns:
                holidays_df = pd.DataFrame({
                    'holiday': 'libur_nasional',
                    'ds': pd.to_datetime(df_hol['Tanggal']),
                    'lower_window': 0, 'upper_window': 0,
                })
                st.success("Data Hari Libur berhasil dimuat.")

        # Tombol Eksekusi
        if st.button("🚀 Jalankan Forecasting", type="primary"):
            with st.spinner("Melatih model Prophet dan menghitung Erlang C..."):
                
                # --- LOGIKA SCRIPT 3: SEASONALITY ---
                model_kwargs = {
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'daily_seasonality': False
                }
                
                if holidays_df is not None:
                    model = Prophet(holidays=holidays_df, **model_kwargs)
                else:
                    model = Prophet(**model_kwargs)
                
                if use_seasonality:
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                
                model.add_regressor('tanggal_satu')
                model.fit(df_historical)

                # --- MEMBUAT PREDIKSI ---
                future_dates = pd.date_range(start=start_forecast, end=end_forecast, freq='D')
                future = pd.DataFrame({'ds': future_dates})
                future['tanggal_satu'] = (future['ds'].dt.day == 1).astype(int)
                
                forecast = model.predict(future)
                forecast['yhat_positive'] = forecast['yhat'].apply(lambda x: math.ceil(max(0, x)))

                # --- MENGHITUNG KEBUTUHAN AGEN ---
                forecast['agents_needed'] = forecast['yhat_positive'].apply(
                    lambda x: calculate_agents_erlang_c(x, aht_seconds, target_sl, target_asa_seconds, shrinkage_rate, hours_per_agent_day)
                )
                forecast['agents_adjusted'] = forecast['agents_needed'].apply(lambda x: math.ceil(x / efficiency_factor))
                forecast['SL_Projection'] = forecast.apply(
                    lambda row: calculate_actual_sl(row['yhat_positive'], row['agents_needed'], aht_seconds, target_asa_seconds, hours_per_agent_day), axis=1
                )

                # --- VISUALISASI ---
                st.write("---")
                st.write("### 📈 Hasil Forecasting Prophet")
                fig1 = model.plot(forecast)
                plt.plot(df_historical['ds'], df_historical['y'], 'k.', label='Data Historis')
                st.pyplot(fig1)

                st.write("### 👥 Kebutuhan Agen Harian (Adjusted)")
                fig_agents, ax = plt.subplots(figsize=(12, 4))
                ax.plot(forecast['ds'], forecast['agents_adjusted'], marker='o', color='skyblue')
                ax.set_title("Perkiraan Kebutuhan Agen Harian")
                ax.grid(True)
                st.pyplot(fig_agents)

                if use_seasonality:
                    st.write("### 🧩 Komponen Tren & Seasonality")
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)

                # --- TABEL HASIL & DOWNLOAD ---
                st.write("### 📑 Tabel Prediksi Harian")
                tabel_tampil = forecast[['ds', 'yhat_positive', 'agents_needed', 'agents_adjusted', 'SL_Projection']].copy()
                tabel_tampil['ds'] = tabel_tampil['ds'].dt.strftime('%Y-%m-%d')
                tabel_tampil['SL_Projection'] = tabel_tampil['SL_Projection'].apply(lambda x: f"{x:.2%}")
                st.dataframe(tabel_tampil, use_container_width=True)

                # --- SIMPAN KE MEMORI UNTUK DOWNLOAD ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    tabel_tampil.to_excel(writer, sheet_name='Daily_Forecast', index=False)
                excel_data = output.getvalue()

                st.download_button(
                    label="📥 Download Hasil ke Excel",
                    data=excel_data,
                    file_name="Forecast_WFM_Result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
else:
    st.info("👋 Selamat datang! Silakan upload file Excel Data Historis Anda di menu sebelah kiri untuk memulai.")
apakah bisa?
