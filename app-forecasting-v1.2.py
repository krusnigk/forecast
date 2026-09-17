import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import pulp
import google.generativeai as genai
import math

# ==========================================
# 1. KONFIGURASI HALAMAN & KEAMANAN
# ==========================================
st.set_page_config(page_title="WFM DSS Pro", layout="wide")
st.title("🚀 AI-Powered WFM Contact Center DSS")

# ==========================================
# 2. FUNGSI INTI WFM & PARSER
# ==========================================
@st.cache_data
def robust_wfm_parser(uploaded_file, expected_target, start_date, end_date):
    """
    Membaca file Excel/CSV dengan Smart Auto-Detect Kolom.
    Tahan terhadap perubahan format date/locale dari Google Drive.
    """
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
    
    # Bersihkan spasi tersembunyi pada header
    df.columns = df.columns.str.strip()
    
    # SMART DETECT: Ambil paksa kolom 1 (Waktu) dan kolom 2 (Metrik)
    col_waktu = df.columns[0]
    col_metrik = df.columns[1]
    
    df = df.rename(columns={col_waktu: 'Datetime', col_metrik: expected_target})
    
    # Cleansing data desimal (ubah koma jadi titik)
    if df[expected_target].dtype == 'object': 
        df[expected_target] = df[expected_target].astype(str).str.replace(',', '.').astype(float)
        
    # Standardisasi format waktu
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', dayfirst=True).dt.round('30min')
    df = df.dropna(subset=['Datetime']) 
    
    # Filtering berdasarkan rentang tanggal
    mask = (df['Datetime'] >= pd.to_datetime(start_date)) & (df['Datetime'] <= pd.to_datetime(end_date) + pd.Timedelta(days=1, seconds=-1))
    df = df[mask].groupby('Datetime')[expected_target].first().reset_index()
    
    # Resample untuk memastikan tidak ada interval 30 menit yang bolong
    return df.set_index('Datetime').resample('30min').asfreq().fillna(0).reset_index()

def calculate_erlang_c(volume, aht, target_sl, target_time):
    """
    Kalkulasi probabilitas antrean Erlang C untuk menentukan agen minimum.
    """
    if volume <= 0: return 0
    # Konversi AHT dan SL ke unit yang sama (detik)
    traffic_intensity = (volume * aht) / 1800 # 1800 detik = 30 menit
    agents = math.ceil(traffic_intensity)
    
    # Iterasi mencari jumlah agen ideal untuk memenuhi SLA
    while True:
        erlang_b = 1.0
        for i in range(1, agents + 1):
            erlang_b = (traffic_intensity * erlang_b) / (i + (traffic_intensity * erlang_b))
        
        erlang_c = erlang_b / (1 - (traffic_intensity / agents) * (1 - erlang_b)) if agents > traffic_intensity else 1.0
        
        sl_achieved = 1 - (erlang_c * math.exp(-(agents - traffic_intensity) * (target_time / aht)))
        if sl_achieved >= target_sl:
            return agents
        agents += 1

# ==========================================
# 3. ANTARMUKA PENGGUNA (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    
    # Input API Key (Aman: karakter disembunyikan)
    api_key_input = st.text_input("🔑 Gemini API Key", type="password", help="Masukkan kunci untuk analitik AI.")
    
    st.divider()
    st.subheader("📅 Rentang Data Historis")
    start_hist = st.date_input("Mulai Historis")
    end_hist = st.date_input("Akhir Historis")
    
    st.divider()
    st.subheader("🎯 Parameter Erlang C")
    target_sl = st.slider("Target Service Level (%)", 50, 100, 80) / 100
    sl_time = st.number_input("SL Threshold (Detik)", value=20)
    default_aht = st.number_input("Default AHT (Detik)", value=150)

# ==========================================
# 4. PROSES UTAMA & EKSEKUSI
# ==========================================
col1, col2 = st.columns(2)
with col1:
    file_cof = st.file_uploader("📂 Upload Data Volume (COF)", type=['xlsx', 'csv'])
with col2:
    file_aht = st.file_uploader("📂 Upload Data AHT (Opsional)", type=['xlsx', 'csv'])

if st.button("🚀 Jalankan Analitik WFM") and file_cof:
    with st.spinner("Memproses data lintas perangkat..."):
        # Eksekusi Parser Tahan Banting
        df_cof = robust_wfm_parser(file_cof, 'COF', start_hist, end_hist)
        
        if file_aht:
            df_aht = robust_wfm_parser(file_aht, 'AHT', start_hist, end_hist)
            df_master = pd.merge(df_cof, df_aht, on='Datetime', how='left').fillna(default_aht)
        else:
            df_master = df_cof.copy()
            df_master['AHT'] = default_aht

        # Forecasting Sederhana dengan Prophet
        df_prophet = df_master.rename(columns={'Datetime': 'ds', 'COF': 'y'})
        m = Prophet(daily_seasonality=True, yearly_seasonality=False)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=48, freq='30min') # Prediksi 1 hari ke depan (48 interval)
        forecast = m.predict(future)
        
        # Ekstrak hasil prediksi
        forecast_today = forecast.tail(48)[['ds', 'yhat']].rename(columns={'ds': 'Interval', 'yhat': 'Prediksi_Volume'})
        forecast_today['Prediksi_Volume'] = np.maximum(0, forecast_today['Prediksi_Volume'].round())
        
        # Kalkulasi Kapasitas (Erlang)
        forecast_today['Kebutuhan_Agen'] = forecast_today.apply(
            lambda row: calculate_erlang_c(row['Prediksi_Volume'], default_aht, target_sl, sl_time), axis=1
        )
        
        st.success("✅ Pemrosesan Data Berhasil!")
        
        st.subheader("📊 Hasil Forecast & Kebutuhan Agen (1 Hari Ke Depan)")
        st.dataframe(forecast_today, use_container_width=True)
        st.line_chart(forecast_today.set_index('Interval')['Prediksi_Volume'])

        # ==========================================
        # 5. INTEGRASI GEMINI AI
        # ==========================================
        if api_key_input:
            st.divider()
            st.subheader("🤖 AI Executive Summary")
            with st.spinner("Gemini AI sedang menyusun analisis strategi..."):
                try:
                    genai.configure(api_key=api_key_input)
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    
                    # Rangkum data untuk AI agar token tidak membengkak
                    total_vol = forecast_today['Prediksi_Volume'].sum()
                    max_agents = forecast_today['Kebutuhan_Agen'].max()
                    peak_hour = forecast_today.loc[forecast_today['Prediksi_Volume'].idxmax(), 'Interval']
                    
                    prompt = f"""
                    Sebagai Senior WFM Analyst, berikan ringkasan eksekutif 2 paragraf berdasarkan data forecast besok:
                    - Total Volume Panggilan: {total_vol}
                    - Puncak Panggilan Terjadi Pada: {peak_hour}
                    - Kebutuhan Agen Maksimal: {max_agents} agen.
                    Berikan rekomendasi penjadwalan singkat untuk jam sibuk tersebut.
                    """
                    response = model.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gagal memuat AI: Pastikan API Key valid. Detail: {e}")
