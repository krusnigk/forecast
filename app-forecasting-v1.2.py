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
def cleanse_data_hw(df, target_col, seasonal_periods=48, threshold=2.0, min_residual=15):
    # PERBAIKAN: Menggunakan .ffill() dan .bfill() untuk kompatibilitas Pandas versi terbaru
    series = df[target_col].ffill().bfill()
    
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated")
    hw_fit = model.fit()
    fitted_values = hw_fit.fittedvalues
    
    residuals = np.abs(series - fitted_values)
    std_dev = np.std(residuals)
    
    # Anomali: melebihi threshold standar deviasi DAN selisih fluktuasinya lebih dari min_residual
    is_anomaly = (residuals > (threshold * std_dev)) & (residuals > min_residual)
    
    cleansed_series = series.copy()
    cleansed_series[is_anomaly] = fitted_values[is_anomaly]
    
    return cleansed_series, is_anomaly

# --- FUNGSI FORECAST PROPHET ---
def run_prophet(df_hist, df_holidays, target_col, start_fcst, end_fcst, use_auto_payday=True):
    df_prophet = pd.DataFrame({
        'ds': df_hist['Datetime'],
        'y': df_hist[f'{target_col}_cleansed']
    })
    
    # Mencegah error pada mode multiplikatif jika ada nilai 0 mutlak
    df_prophet['y'] = df_prophet['y'].replace(0, 0.01)
    
    holidays_list = []
    holiday_dates = []
    
    # 1. Masukkan Hari Libur Nasional
    if df_holidays is not None and not df_holidays.empty:
        if 'Tanggal' in df_holidays.columns:
            h_df = pd.DataFrame({
                'holiday': 'libur_nasional',
                'ds': pd.to_datetime(df_holidays['Tanggal']),
                'lower_window': 0,
                'upper_window': 0
            })
            holidays_list.append(h_df)
            holiday_dates = pd.to_datetime(df_holidays['Tanggal']).dt.normalize().tolist()
            
    # 2. Masukkan Logika Payday Otomatis
    if use_auto_payday:
        start_date = df_prophet['ds'].min()
        end_date = pd.to_datetime(end_fcst)
        
        dr = pd.date_range(start=start_date, end=end_date)
        months = dr.to_period('M').unique()
        
        paydays = []
        for m in months:
            # Tanggal 1
            dt_1 = pd.Timestamp(year=m.year, month=m.month, day=1)
            paydays.append(dt_1)
            
            # Tanggal 25 (Digeser jika Sabtu, Minggu, atau Hari Libur Nasional)
            dt_25 = pd.Timestamp(year=m.year, month=m.month, day=25)
            while dt_25.weekday() >= 5 or dt_25 in holiday_dates:
                dt_25 -= pd.Timedelta(days=1)
            paydays.append(dt_25)
            
        p_df = pd.DataFrame({
            'holiday': 'payday',
            'ds': list(set(paydays)),
            'lower_window': 0,
            'upper_window': 0
        })
        holidays_list.append(p_df)
        
    final_holidays = pd.concat(holidays_list, ignore_index=True) if holidays_list else None
        
    model = Prophet(holidays=final_holidays, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False, seasonality_mode='multiplicative')
    model.fit(df_prophet)
    
    last_hist_date = df_prophet['ds'].max()
    end_fcst_dt = pd.to_datetime(end_fcst) + pd.Timedelta(days=1, minutes=-30)
    
    if end_fcst_dt > last_hist_date:
        delta = end_fcst_dt - last_hist_date
        periods = int(delta.total_seconds() / 1800)
    else:
        periods = 0
        
    future = model.make_future_dataframe(periods=periods, freq='30T', include_history=True)
    forecast = model.predict(future)
    
    historical_forecast = forecast.iloc[:len(df_prophet)]
    mape = mean_absolute_percentage_error(df_prophet['y'], historical_forecast['yhat']) * 100
    
    start_fcst_dt = pd.to_datetime(start_fcst)
    future_forecast = forecast[(forecast['ds'] >= start_fcst_dt) & (forecast['ds'] <= end_fcst_dt)][['ds', 'yhat']]
    future_forecast.rename(columns={'ds': 'Datetime', 'yhat': f'{target_col}_forecast'}, inplace=True)
    future_forecast[f'{target_col}_forecast'] = future_forecast[f'{target_col}_forecast'].clip(lower=0)
    
    return future_forecast, mape

# --- UI SIDEBAR ---
st.sidebar.header("📂 1. Upload Database")
st.sidebar.markdown("*(Pastikan ada kolom **Datetime** untuk Interval & **Tanggal** untuk Hari Libur)*")
file_cof = st.sidebar.file_uploader("Upload Data COF (Interval 30 Min)", type=['csv', 'xlsx'])
file_aht = st.sidebar.file_uploader("Upload Data AHT (Interval 30 Min)", type=['csv', 'xlsx'])
file_holidays = st.sidebar.file_uploader("Upload Data Libur Nasional (Opsional)", type=['csv', 'xlsx'])

st.sidebar.header("⚙️ 2. Konfigurasi Erlang C")
shrinkage = st.sidebar.number_input("Shrinkage (%)", min_value=0.0, max_value=100.0, value=30.0) / 100
max_wait_time = st.sidebar.number_input("Max Waiting Time (Detik)", value=20)
work_hours = st.sidebar.number_input("Jam Kerja per Hari", value=8)
work_days = st.sidebar.number_input("Hari Kerja/Agen/Bulan", value=22)

st.sidebar.header("📅 3. Konfigurasi Tanggal")
st.sidebar.markdown("**Data Historis (Training)**")
start_hist = st.sidebar.date_input("Mulai Data Historis", pd.to_datetime('2024-02-01'))
end_hist = st.sidebar.date_input("Akhir Data Historis", pd.to_datetime('2026-05-25'))

st.sidebar.markdown("**Target Forecast**")
start_forecast = st.sidebar.date_input("Mulai Forecast", pd.to_datetime('2026-06-01'))
end_forecast = st.sidebar.date_input("Akhir Forecast", pd.to_datetime('2026-06-30'))

use_payday = st.sidebar.checkbox("💰 Aktifkan Auto-Payday (Tgl 1 & 25)", value=True, help="Menandai otomatis tgl 1 & 25 sbg payday. Jika libur/weekend, mundur ke hari kerja sebelumnya.")

# --- PROSES UTAMA ---
if st.button("Jalankan Forecast & Kalkulasi", type="primary"):
    if file_cof and file_aht:
        with st.spinner("Memproses cleansing, payday, dan forecasting. Mohon tunggu..."):
            df_cof = pd.read_csv(file_cof) if file_cof.name.endswith('csv') else pd.read_excel(file_cof)
            df_aht = pd.read_csv(file_aht) if file_aht.name.endswith('csv') else pd.read_excel(file_aht)
            
            df_holidays = None
            if file_holidays:
                df_holidays = pd.read_csv(file_holidays) if file_holidays.name.endswith('csv') else pd.read_excel(file_holidays)
            
            df_cof['Datetime'] = pd.to_datetime(df_cof['Datetime'])
            df_aht['Datetime'] = pd.to_datetime(df_aht['Datetime'])
            
            df_cof = df_cof[(df_cof['Datetime'] >= pd.to_datetime(start_hist)) & (df_cof['Datetime'] <= pd.to_datetime(end_hist) + pd.Timedelta(days=1, seconds=-1))].copy()
            df_aht = df_aht[(df_aht['Datetime'] >= pd.to_datetime(start_hist)) & (df_aht['Datetime'] <= pd.to_datetime(end_hist) + pd.Timedelta(days=1, seconds=-1))].copy()
            
            # Cleansing
            df_cof['COF_cleansed'], _ = cleanse_data_hw(df_cof, 'COF', min_residual=15)
            df_aht['AHT_cleansed'], _ = cleanse_data_hw(df_aht, 'AHT', min_residual=50)
            
            # Forecasting
            forecast_cof, mape_cof = run_prophet(df_cof, df_holidays, 'COF', start_forecast, end_forecast, use_auto_payday=use_payday)
            forecast_aht, mape_aht = run_prophet(df_aht, df_holidays, 'AHT', start_forecast, end_forecast, use_auto_payday=use_payday)
            
            df_result = pd.merge(forecast_cof, forecast_aht, on='Datetime')
            
            # Erlang C Kalkulasi
            df_result['Base_Agent_Needed'] = 0
            df_result['Projected_Wait_Time'] = 0.0
            
            for index, row in df_result.iterrows():
                agents, wait_time = calculate_agents_erlang(row['COF_forecast'], row['AHT_forecast'], max_wait_time)
                df_result.at[index, 'Base_Agent_Needed'] = agents
                df_result.at[index, 'Projected_Wait_Time'] = wait_time
                
            df_result['Agent_Needed_Adjust'] = np.ceil(df_result['Base_Agent_Needed'] / (1 - shrinkage))
            
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
                c3.metric("Estimasi Total Headcount Bulanan", total_monthly_headcount)
                
                st.write("**Detail Interval Forecast & Agent**")
                st.dataframe(df_result[['Datetime', 'COF_forecast', 'AHT_forecast', 'Base_Agent_Needed', 'Projected_Wait_Time', 'Agent_Needed_Adjust']], use_container_width=True)

    else:
        st.error("Mohon unggah file COF dan AHT terlebih dahulu.")
