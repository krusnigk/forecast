import streamlit as st
import pandas as pd
import numpy as np
import math
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="WFM Forecast & Capacity Planner", layout="wide")
st.title("WFM Forecast & Capacity Planning Tool")

# --- FUNGSI ERLANG C ITERATIF ---
def erlang_c_prob(agents, traffic):
    if agents <= traffic:
        return 1.0
    erlang_b_inv = 1.0
    for i in range(1, int(agents) + 1):
        erlang_b_inv = 1.0 + erlang_b_inv * i / traffic
    erlang_b = 1.0 / erlang_b_inv
    erlang_c = erlang_b / (1.0 - (traffic / agents) * (1.0 - erlang_b))
    return max(0.0, min(1.0, erlang_c))

def calculate_agents_erlang(cof, aht_seconds, max_wait_time):
    if pd.isna(cof) or pd.isna(aht_seconds) or cof <= 0:
        return 0, 0.0
    
    interval_seconds = 1800 # 30 menit
    traffic = (cof * aht_seconds) / interval_seconds
    agents = math.ceil(traffic)
    
    while True:
        prob_wait = erlang_c_prob(agents, traffic)
        if agents > traffic:
            asa = prob_wait * (aht_seconds / (agents - traffic))
        else:
            asa = float('inf')
            
        if asa <= max_wait_time:
            break
        agents += 1
        
    return agents, asa

# --- FUNGSI CLEANSING HOLT-WINTERS ---
# Tetap menggunakan min_residual agar jam malam tidak terpotong habis
def cleanse_data_hw(df, target_col, seasonal_periods=48, threshold=2.0, min_residual=15):
    series = df[target_col].fillna(method='ffill').fillna(method='bfill')
    
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated")
    hw_fit = model.fit()
    fitted_values = hw_fit.fittedvalues
    
    residuals = np.abs(series - fitted_values)
    std_dev = np.std(residuals)
    
    # Anomali: harus lebih dari std_dev threshold DAN selisihnya > min_residual
    is_anomaly = (residuals > (threshold * std_dev)) & (residuals > min_residual)
    
    cleansed_series = series.copy()
    cleansed_series[is_anomaly] = fitted_values[is_anomaly]
    
    return cleansed_series, is_anomaly

# --- FUNGSI FORECAST PROPHET ---
def run_prophet(df_hist, df_payday, target_col, start_fcst, end_fcst):
    df_prophet = pd.DataFrame({
        'ds': df_hist['Datetime'],
        'y': df_hist[f'{target_col}_cleansed']
    })
    
    # Mencegah error pada mode multiplikatif jika ada nilai 0
    df_prophet['y'] = df_prophet['y'].replace(0, 0.01)
    
    holidays = None
    if df_payday is not None and not df_payday.empty:
        holidays = pd.DataFrame({
            'holiday': 'payday',
            'ds': pd.to_datetime(df_payday['Date']),
            'lower_window': 0,
            'upper_window': 0
        })
        
    # Tetap menggunakan mode multiplikatif
    model = Prophet(holidays=holidays, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False, seasonality_mode='multiplicative')
    model.fit(df_prophet)
    
    # Menghitung durasi ke masa depan (dari akhir data historis sampai akhir tanggal forecast)
    last_hist_date = df_prophet['ds'].max()
    end_fcst_dt = pd.to_datetime(end_fcst) + pd.Timedelta(days=1, minutes=-30) # Sampai 23:30 di hari terakhir forecast
    
    if end_fcst_dt > last_hist_date:
        delta = end_fcst_dt - last_hist_date
        periods = int(delta.total_seconds() / 1800)
    else:
        periods = 0
        
    future = model.make_future_dataframe(periods=periods, freq='30T', include_history=True)
    forecast = model.predict(future)
    
    # Hitung MAPE di area historis
    historical_forecast = forecast.iloc[:len(df_prophet)]
    mape = mean_absolute_percentage_error(df_prophet['y'], historical_forecast['yhat']) * 100
    
    # Ambil nilai forecast hanya pada rentang tanggal start_fcst s/d end_fcst
    start_fcst_dt = pd.to_datetime(start_fcst)
    future_forecast = forecast[(forecast['ds'] >= start_fcst_dt) & (forecast['ds'] <= end_fcst_dt)][['ds', 'yhat']]
    future_forecast.rename(columns={'ds': 'Datetime', 'yhat': f'{target_col}_forecast'}, inplace=True)
    future_forecast[f'{target_col}_forecast'] = future_forecast[f'{target_col}_forecast'].clip(lower=0)
    
    return future_forecast, mape

# --- UI SIDEBAR ---
st.sidebar.header("📂 1. Upload Database")
st.sidebar.markdown("*(Pastikan ada kolom **Datetime** untuk Interval & **Date** untuk Pay Day)*")
file_cof = st.sidebar.file_uploader("Upload Data COF (Interval 30 Min)", type=['csv', 'xlsx'])
file_aht = st.sidebar.file_uploader("Upload Data AHT (Interval 30 Min)", type=['csv', 'xlsx'])
file_payday = st.sidebar.file_uploader("Upload Data Pay Day (Opsional)", type=['csv', 'xlsx'])

st.sidebar.header("⚙️ 2. Konfigurasi Erlang C")
shrinkage = st.sidebar.number_input("Shrinkage (%)", min_value=0.0, max_value=100.0, value=30.0) / 100
max_wait_time = st.sidebar.number_input("Max Waiting Time (Detik)", value=20)
work_hours = st.sidebar.number_input("Jam Kerja per Hari", value=8)
work_days = st.sidebar.number_input("Hari Kerja/Agen/Bulan", value=22)

# DIKEMBALIKAN: Input Tanggal
st.sidebar.header("📅 3. Konfigurasi Tanggal")
st.sidebar.markdown("**Data Historis (Training)**")
start_hist = st.sidebar.date_input("Mulai Data Historis", pd.to_datetime('2024-02-01'))
end_hist = st.sidebar.date_input("Akhir Data Historis", pd.to_datetime('2026-05-25'))

st.sidebar.markdown("**Target Forecast**")
start_forecast = st.sidebar.date_input("Mulai Forecast", pd.to_datetime('2026-06-01'))
end_forecast = st.sidebar.date_input("Akhir Forecast", pd.to_datetime('2026-06-30'))

# --- PROSES UTAMA ---
if st.button("Jalankan Forecast & Kalkulasi", type="primary"):
    if file_cof and file_aht:
        with st.spinner("Memproses cleansing dan forecasting, mohon tunggu..."):
            # 1. Baca Data
            df_cof = pd.read_csv(file_cof) if file_cof.name.endswith('csv') else pd.read_excel(file_cof)
            df_aht = pd.read_csv(file_aht) if file_aht.name.endswith('csv') else pd.read_excel(file_aht)
            
            df_payday = None
            if file_payday:
                df_payday = pd.read_csv(file_payday) if file_payday.name.endswith('csv') else pd.read_excel(file_payday)
            
            df_cof['Datetime'] = pd.to_datetime(df_cof['Datetime'])
            df_aht['Datetime'] = pd.to_datetime(df_aht['Datetime'])
            
            # Filter Data Historis berdasarkan input kalender user
            # Ditambah 1 hari - 1 detik agar meng-cover sampai 23:59 di hari terakhir historis
            df_cof = df_cof[(df_cof['Datetime'] >= pd.to_datetime(start_hist)) & (df_cof['Datetime'] <= pd.to_datetime(end_hist) + pd.Timedelta(days=1, seconds=-1))].copy()
            df_aht = df_aht[(df_aht['Datetime'] >= pd.to_datetime(start_hist)) & (df_aht['Datetime'] <= pd.to_datetime(end_hist) + pd.Timedelta(days=1, seconds=-1))].copy()
            
            # 2. Cleansing Data (Holt-Winters)
            df_cof['COF_cleansed'], _ = cleanse_data_hw(df_cof, 'COF', min_residual=15)
            df_aht['AHT_cleansed'], _ = cleanse_data_hw(df_aht, 'AHT', min_residual=50)
            
            # 3. Forecasting Prophet menggunakan kalender input
            forecast_cof, mape_cof = run_prophet(df_cof, df_payday, 'COF', start_forecast, end_forecast)
            forecast_aht, mape_aht = run_prophet(df_aht, df_payday, 'AHT', start_forecast, end_forecast)
            
            # Gabungkan Hasil Forecast
            df_result = pd.merge(forecast_cof, forecast_aht, on='Datetime')
            
            # 4. Kalkulasi Erlang C per Interval
            df_result['Base_Agent_Needed'] = 0
            df_result['Projected_Wait_Time'] = 0.0
            
            for index, row in df_result.iterrows():
                agents, wait_time = calculate_agents_erlang(row['COF_forecast'], row['AHT_forecast'], max_wait_time)
                df_result.at[index, 'Base_Agent_Needed'] = agents
                df_result.at[index, 'Projected_Wait_Time'] = wait_time
                
            # Hitung Agent Adjusted (Dengan Shrinkage)
            df_result['Agent_Needed_Adjust'] = np.ceil(df_result['Base_Agent_Needed'] / (1 - shrinkage))
            
            # Kalkulasi Headcount Bulanan menggunakan selisih hari forecast
            df_result['Date'] = df_result['Datetime'].dt.date
            max_agents_per_day = df_result.groupby('Date')['Agent_Needed_Adjust'].max()
            avg_daily_headcount_needed = max_agents_per_day.mean()
            
            total_hari_forecast = (pd.to_datetime(end_forecast) - pd.to_datetime(start_forecast)).days + 1
            total_monthly_headcount = math.ceil((avg_daily_headcount_needed * total_hari_forecast) / work_days)

            # --- TAMPILAN HASIL (UI) ---
            st.success("Proses Selesai!")
            
            tab1, tab2, tab3 = st.tabs(["📊 Forecast & Cleansing", "🎯 Akurasi (MAPE)", "👥 Kebutuhan Agent"])
            
            with tab1:
                st.subheader("Prediksi COF (Call Offered)")
                st.line_chart(df_result.set_index('Datetime')['COF_forecast'])
                st.subheader("Prediksi AHT (Average Handle Time)")
                st.line_chart(df_result.set_index('Datetime')['AHT_forecast'])
                
            with tab2:
                col1, col2 = st.columns(2)
                col1.metric(label="MAPE COF (Historical Fit)", value=f"{mape_cof:.2f}%")
                col2.metric(label="MAPE AHT (Historical Fit)", value=f"{mape_aht:.2f}%")
                if mape_cof > 15 or mape_aht > 15:
                    st.warning("⚠️ Nilai MAPE di atas 15%. Pertimbangkan untuk menambah/memperbaiki data historis.")
                else:
                    st.success("✅ Akurasi model dalam batas yang sangat baik.")
                    
            with tab3:
                st.subheader("Ringkasan Kapasitas (Kalkulasi Erlang C)")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Max Kebutuhan Agent per Interval", int(df_result['Agent_Needed_Adjust'].max()))
                c2.metric("Rata-rata Wait Time Proyeksi (Detik)", f"{df_result['Projected_Wait_Time'].mean():.2f}")
                c3.metric("Estimasi Total Headcount Bulanan", total_monthly_headcount, help=f"Dihitung berdasarkan forecast selama {total_hari_forecast} hari dibagi {work_days} hari kerja efektif.")
                
                st.write("**Detail Interval Forecast & Agent**")
                st.dataframe(df_result[['Datetime', 'COF_forecast', 'AHT_forecast', 'Base_Agent_Needed', 'Projected_Wait_Time', 'Agent_Needed_Adjust']], use_container_width=True)

    else:
        st.error("Mohon unggah file COF dan AHT terlebih dahulu.")
