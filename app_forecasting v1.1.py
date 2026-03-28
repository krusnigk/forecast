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

def calculate_actual_sl(daily_forecast_value, agents_needed_for_day, aht_seconds, target_asa_seconds, shrinkage_rate, hours_per_agent_day):
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

st.sidebar.header("📅 3. Konfigurasi Tanggal Forecast")
st.sidebar.markdown("**Data Historis (Training)**")
start_hist = st.sidebar.date_input("Mulai Data Historis", pd.to_datetime('2024-12-01'))
end_hist = st.sidebar.date_input("Akhir Data Historis", pd.to_datetime('2026-02-28'))

st.sidebar.markdown("**Target Forecast**")
start_forecast = st.sidebar.date_input("Mulai Forecast", pd.to_datetime('2026-03-01'))
end_forecast = st.sidebar.date_input("Akhir Forecast", pd.to_datetime('2026-03-31'))

# FITUR BARU: Konfigurasi Hari Kalender & Hari Kerja
st.sidebar.header("🗓️ 4. Parameter Bulanan (Headcount)")
st.sidebar.markdown("Penentu kalkulasi total agen bulanan")
hari_kalender = st.sidebar.number_input("Jumlah Hari Kalender", min_value=1, value=31, help="Total hari dalam bulan berjalan")
hari_kerja = st.sidebar.number_input("Jumlah Hari Kerja (HKE)", min_value=1, value=22, help="Jumlah hari kerja efektif 1 agen dalam sebulan")

# ==========================================
# AREA UTAMA: PENGATURAN MODEL & ANOMALI
# ==========================================
st.write("### 🛠️ Pengaturan Model Prophet")

with st.expander("⚠️ Pengaturan Filter Anomali (Ubah Tanggal Anomali)", expanded=True):
    use_anomaly_filter = st.checkbox("✅ Aktifkan Filter Anomali", value=True, help="Menghapus data anomali pada rentang tanggal tertentu agar tidak merusak tren model.")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Anomali Periode 1**")
        anomali_1_start = st.date_input("Mulai Anomali 1", pd.to_datetime('2025-03-01'))
        anomali_1_end = st.date_input("Akhir Anomali 1", pd.to_datetime('2025-03-31'))
    with col_a2:
        st.markdown("**Anomali Periode 2**")
        anomali_2_start = st.date_input("Mulai Anomali 2", pd.to_datetime('2026-02-19'))
        anomali_2_end = st.date_input("Akhir Anomali 2", pd.to_datetime('2026-03-22'))

use_seasonality = st.checkbox("🌊 Aktifkan Seasonality Bulanan (Siklus ~30.5 Hari)", value=True)

# ==========================================
# PROSES UTAMA
# ==========================================
if uploaded_data is not None:
    try:
        # Membaca data
        df = pd.read_excel(uploaded_data)
        df = df.rename(columns={'date': 'ds', 'cof': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
        
        # Menerapkan Filter Tanggal Data Historis dari UI
        df_historical = df[(df['ds'] >= pd.to_datetime(start_hist)) & (df['ds'] <= pd.to_datetime(end_hist))].copy()

        # Tambahan Regressor Tanggal 1
        df_historical['tanggal_satu'] = (df_historical['ds'].dt.day == 1).astype(int)

        # Menerapkan Filter Anomali dari UI
        if use_anomaly_filter:
            anomali_1 = (df_historical['ds'] >= pd.to_datetime(anomali_1_start)) & (df_historical['ds'] <= pd.to_datetime(anomali_1_end))
            anomali_2 = (df_historical['ds'] >= pd.to_datetime(anomali_2_start)) & (df_historical['ds'] <= pd.to_datetime(anomali_2_end))
            kondisi_anomali = anomali_1 | anomali_2
            jumlah_anomali = kondisi_anomali.sum()
            df_historical.loc[kondisi_anomali, 'y'] = np.nan
            st.info(f"Filter Anomali Aktif: {jumlah_anomali} baris data pada histori dikosongkan (diabaikan) sebelum training.")

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
                
                # --- PELATIHAN MODEL ---
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

                # LOGIKA PENYESUAIAN HARI MINGGU (10% Lebih Rendah dari Sabtu)
                for i in range(1, len(forecast)):
                    if forecast.loc[i, 'ds'].weekday() == 6: 
                        if forecast.loc[i-1, 'ds'].weekday() == 5:
                            sabtu_volume = forecast.loc[i-1, 'yhat_positive']
                            forecast.loc[i, 'yhat_positive'] = math.ceil(sabtu_volume * 0.90)

                # --- MENGHITUNG KEBUTUHAN AGEN HARIAN ---
                forecast['agents_needed'] = forecast['yhat_positive'].apply(
                    lambda x: calculate_agents_erlang_c(x, aht_seconds, target_sl, target_asa_seconds, shrinkage_rate, hours_per_agent_day)
                )
                forecast['agents_adjusted'] = forecast['agents_needed'].apply(lambda x: math.ceil(x / efficiency_factor))
                
                forecast['SL_Projection'] = forecast.apply(
                    lambda row: calculate_actual_sl(
                        row['yhat_positive'], 
                        row['agents_needed'], 
                        aht_seconds, 
                        target_asa_seconds, 
                        shrinkage_rate, 
                        hours_per_agent_day
                    ), axis=1
                )

                # FITUR BARU: KALKULASI HEADCOUNT BULANAN
                total_shift_harian = forecast['agents_adjusted'].sum()
                kebutuhan_headcount_bulanan = math.ceil(total_shift_harian / hari_kerja)

                # --- TAMPILAN RINGKASAN KEBUTUHAN BULANAN ---
                st.write("---")
                st.write("### 🎯 Ringkasan Kebutuhan Headcount Bulanan")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Hari Kalender Input", f"{hari_kalender} Hari")
                col_m2.metric("Hari Kerja/Agen Input", f"{hari_kerja} Hari")
                col_m3.metric("Total Kebutuhan Harian", f"{total_shift_harian} Shift", help="Akumulasi seluruh agen yang dibutuhkan setiap harinya selama periode forecast")
                col_m4.metric("Kebutuhan Agen Bulanan", f"{kebutuhan_headcount_bulanan} Orang", help=f"Dihitung dari: {total_shift_harian} dibagi {hari_kerja}")

                st.success(f"**Kesimpulan:** Berdasarkan total beban kerja sebanyak **{total_shift_harian} shift** dalam periode forecast, dan asumsi 1 agen bekerja efektif **{hari_kerja} hari** dalam sebulan, Anda membutuhkan total **{kebutuhan_headcount_bulanan} Agen** untuk memenuhi target SL.")

                # --- VISUALISASI ---
                st.write("---")
                st.write("### 📈 Hasil Forecasting Prophet")
                fig1 = model.plot(forecast)
                plt.plot(df_historical['ds'], df_historical['y'], 'k.', label='Data Historis (Dipakai)')
                st.pyplot(fig1)

                st.write("### 👥 Kebutuhan Agen Harian (Adjusted)")
                fig_agents, ax = plt.subplots(figsize=(12, 4))
                ax.plot(forecast['ds'], forecast['agents_adjusted'], marker='o', color='skyblue')
                ax.set_title("Perkiraan Kebutuhan Agen Harian")
                ax.grid(True)
                st.pyplot(fig_agents)

                # --- TABEL HASIL & DOWNLOAD ---
                st.write("### 📑 Tabel Prediksi Harian")
                tabel_tampil = forecast[['ds', 'yhat_positive', 'agents_needed', 'agents_adjusted', 'SL_Projection']].copy()
                tabel_tampil.columns = ['Tanggal', 'Prediksi COF (Adjusted)', 'Agen Dibutuhkan', 'Agen + Shrinkage/Efisiensi', 'Proyeksi SL']
                
                tabel_tampil['Hari'] = tabel_tampil['Tanggal'].dt.day_name()
                tabel_tampil['Tanggal'] = tabel_tampil['Tanggal'].dt.strftime('%Y-%m-%d')
                tabel_tampil['Proyeksi SL'] = tabel_tampil['Proyeksi SL'].apply(lambda x: f"{x:.2%}")
                
                tabel_tampil = tabel_tampil[['Tanggal', 'Hari', 'Prediksi COF (Adjusted)', 'Agen Dibutuhkan', 'Agen + Shrinkage/Efisiensi', 'Proyeksi SL']]
                
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
