import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import math
import io

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Forecasting & Workforce Planning", layout="wide")

# --- Fungsi Erlang C (Stabilitas Numerik) ---
def erlang_c(A, N):
    if A <= 0: return 0.0
    if N <= A: return 1.0
    log_sum_terms = -float('inf')
    for k in range(N):
        try:
            term_log = k * math.log(A) - math.lgamma(k + 1)
            log_sum_terms = np.logaddexp(log_sum_terms, term_log)
        except ValueError: pass
    try:
        log_numerator = (N * math.log(A) - math.lgamma(N + 1)) + (math.log(N) - math.log(N - A))
    except (ValueError, OverflowError): return 1.0 
    log_denominator = np.logaddexp(log_sum_terms, log_numerator)
    return max(0.0, min(1.0, math.exp(log_numerator - log_denominator)))

def calculate_agents_erlang_c(daily_calls, aht_seconds, target_sl, target_asa_seconds, shrinkage_rate, hours_per_agent_day):
    if daily_calls <= 0 or hours_per_agent_day <= 0: return 0
    workload_in_erlang = (daily_calls * aht_seconds) / 3600
    A_hourly = ((daily_calls / hours_per_agent_day) * aht_seconds) / 3600 
    min_agents_after_shrinkage = math.ceil((workload_in_erlang / hours_per_agent_day) / (1 - shrinkage_rate))
    
    found_agents = max(1, min_agents_after_shrinkage)
    for _ in range(200):
        N_effective = max(1, math.ceil(found_agents * (1 - shrinkage_rate)))
        if A_hourly <= 0: return math.ceil(found_agents)
        prob_delay = 1.0 if N_effective <= A_hourly else erlang_c(A_hourly, N_effective)
        current_sl = 0.0 if N_effective - A_hourly <= 0 else 1 - prob_delay * math.exp(-((N_effective - A_hourly) * target_asa_seconds) / aht_seconds)
        if current_sl >= target_sl: return math.ceil(found_agents)
        found_agents += 1
    return math.ceil(found_agents)

def calculate_actual_sl(daily_forecast_value, agents_needed_for_day, aht_seconds, target_asa_seconds, shrinkage_rate, hours_per_agent_day):
    if daily_forecast_value <= 0 or agents_needed_for_day <= 0: return 1.0 
    N_effective = max(1, math.ceil(agents_needed_for_day * (1 - shrinkage_rate)))
    A_hourly = ((daily_forecast_value / hours_per_agent_day) * aht_seconds) / 3600
    if A_hourly <= 0: return 1.0 
    prob_delay = 1.0 if N_effective <= A_hourly else erlang_c(A_hourly, N_effective)
    return max(0.0, min(1.0, 1 - prob_delay * math.exp(-((N_effective - A_hourly) * target_asa_seconds) / aht_seconds))) if N_effective > A_hourly else 0.0

# --- SIDEBAR: KONFIGURASI ---
st.sidebar.header("📋 Parameter Konfigurasi")

# Upload Files
uploaded_data = st.sidebar.file_uploader("Unggah Data COF (Excel)", type=["xlsx"])
uploaded_holidays = st.sidebar.file_uploader("Unggah Database Libur (Optional)", type=["xlsx"])

# Date Settings
st.sidebar.subheader("📅 Rentang Tanggal")
h_start = st.sidebar.date_input("Mulai Historis", value=pd.to_datetime('2025-07-01'))
h_end = st.sidebar.date_input("Akhir Historis", value=pd.to_datetime('2026-01-11'))
f_start = st.sidebar.date_input("Mulai Forecast", value=pd.to_datetime('2026-01-12'))
f_end = st.sidebar.date_input("Akhir Forecast", value=pd.to_datetime('2026-01-31'))

# Erlang Settings
st.sidebar.subheader("🎧 Parameter Erlang C")
aht = st.sidebar.number_input("AHT (detik)", value=440)
target_sl = st.sidebar.slider("Target Service Level", 0.0, 1.0, 0.90)
target_asa = st.sidebar.number_input("Target ASA (detik)", value=20)
shrinkage = st.sidebar.slider("Shrinkage Rate", 0.0, 1.0, 0.22)
efficiency = st.sidebar.slider("Faktor Efisiensi", 0.0, 1.0, 0.85)
work_hours = st.sidebar.number_input("Jam Kerja/Hari", value=8)

# Parameter Baru: Jumlah Hari Kerja
st.sidebar.subheader("🗓️ Parameter Bulanan")
working_days_per_month = st.sidebar.number_input("Jumlah Hari Kerja/Bulan", value=22, min_value=1, help="Digunakan sebagai pembagi untuk menentukan headcount bulanan")

# --- MAIN PAGE ---
st.title("📈 Call Volume Forecasting & Agent Calculator by Krusnigk")

if uploaded_data:
    # 1. Load Data
    df = pd.read_excel(uploaded_data)
    df = df.rename(columns={'date': 'ds', 'cof': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df_historical = df[(df['ds'] >= pd.to_datetime(h_start)) & (df['ds'] <= pd.to_datetime(h_end))].copy()
    df_historical['tanggal_satu'] = (df_historical['ds'].dt.day == 1).astype(int)

    # 2. Load Holidays
    holidays_df = None
    if uploaded_holidays:
        df_h = pd.read_excel(uploaded_holidays)
        holidays_df = pd.DataFrame({
            'holiday': 'libur_nasional',
            'ds': pd.to_datetime(df_h['Tanggal']),
            'lower_window': 0, 'upper_window': 0,
        })

    # 3. Model Training
    with st.spinner('Sedang melatih model peramalan...'):
        model = Prophet(holidays=holidays_df) if holidays_df is not None else Prophet()
        model.add_regressor('tanggal_satu')
        model.fit(df_historical)

        # 4. Forecasting
        future_dates = pd.date_range(start=f_start, end=f_end, freq='D')
        future = pd.DataFrame({'ds': future_dates})
        future['tanggal_satu'] = (future['ds'].dt.day == 1).astype(int)
        forecast = model.predict(future)

        # 5. Calculations
        forecast['yhat_positive'] = forecast['yhat'].apply(lambda x: math.ceil(max(0, x)))
        forecast['agents_needed'] = forecast['yhat_positive'].apply(
            lambda x: calculate_agents_erlang_c(x, aht, target_sl, target_asa, shrinkage, work_hours)
        )
        forecast['agents_needed_adj'] = forecast['agents_needed'].apply(lambda x: math.ceil(x / efficiency))
        forecast['SL_Projection'] = forecast.apply(
            lambda row: calculate_actual_sl(row['yhat_positive'], row['agents_needed'], aht, target_asa, shrinkage, work_hours), axis=1
        )

    # --- 6. Tampilan Hasil ---
    
    # Bagian 6a: Metrik Summary (Kebutuhan Bulanan Berdasarkan Input Hari Kerja)
    st.subheader("🗓️ Ringkasan Kebutuhan Bulanan")
    
    total_hari_dalam_forecast = len(forecast)
    total_kebutuhan_akumulatif = forecast['agents_needed_adj'].sum()
    
    # RUMUS: Total kebutuhan harian akumulatif / Jumlah Hari Kerja yang diinput di sidebar
    kebutuhan_bulanan_final = math.ceil(total_kebutuhan_akumulatif / working_days_per_month)
    
    avg_volume = math.ceil(forecast['yhat_positive'].mean())

    m1, m2, m3 = st.columns(3)
    m1.metric("Rentang Forecast", f"{total_hari_dalam_forecast} Hari Kalender")
    m2.metric("Hari Kerja (Pembagi)", f"{working_days_per_month} Hari")
    m3.metric("Kebutuhan Agen Bulanan", f"{kebutuhan_bulanan_final} Orang", 
              help=f"Dihitung dari: {total_kebutuhan_akumulatif} (total beban agen) / {working_days_per_month} hari kerja")

    st.info(f"💡 **Info:** Berdasarkan total beban kerja selama {total_hari_dalam_forecast} hari, Anda membutuhkan rata-rata **{kebutuhan_bulanan_final} agen** untuk mencover operasional dengan asumsi {working_days_per_month} hari kerja efektif per agen.")
    st.markdown("---")

    # Bagian 6b: Grafik
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Grafik Peramalan Volume")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
    
    with col2:
        st.subheader("👥 Grafik Kebutuhan Agen Harian")
        fig2, ax2 = plt.subplots()
        ax2.plot(forecast['ds'], forecast['agents_needed_adj'], marker='o', color='skyblue', label='Agen Dibutuhkan')
        ax2.set_ylabel("Jumlah Agen")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # Bagian 6c: Tabel Detail
    st.subheader("📋 Tabel Hasil Prediksi Detail")
    display_df = forecast[['ds', 'yhat_positive', 'agents_needed_adj', 'SL_Projection']].copy()
    display_df.columns = ['Tanggal', 'Prediksi Volume', 'Kebutuhan Agen (Adj)', 'Proyeksi SL']
    display_df['Proyeksi SL'] = display_df['Proyeksi SL'].map('{:.2%}'.format)
    st.dataframe(display_df, use_container_width=True)

    # 7. Download Button
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        display_df.to_excel(writer, index=False, sheet_name='Forecast_Result')
    
    st.download_button(
        label="📥 Download Hasil Excel",
        data=output.getvalue(),
        file_name=f"Forecast_Result_{f_start}_to_{f_end}.xlsx",
        mime="application/vnd.ms-excel"
    )

else:
    st.info("👋 Silakan unggah file Excel data historis di sidebar untuk memulai.")