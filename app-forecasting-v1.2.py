import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime
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
    
    interval_seconds = 1800 
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

# --- FUNGSI CLEANSING HOLT-WINTERS (FIXED PANDAS ERROR) ---
def cleanse_data_hw(df, target_col, seasonal_periods=48, threshold=2.0):
    # Menggunakan ffill() dan bfill() sesuai standar Pandas terbaru
    series = df[target_col].ffill().bfill()
    
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated")
    hw_fit = model.fit()
    fitted_values = hw_fit.fittedvalues
    
    residuals = np.abs(series - fitted_values)
    std_dev = np.std(residuals)
    is_anomaly = residuals > (threshold * std_dev)
    
    cleansed_series = series.copy()
    cleansed_series[is_anomaly] = fitted_values[is_anomaly]
    
    return cleansed_series, is_anomaly

# --- FUNGSI FORECAST PROPHET (UPDATE CUSTOM DATES) ---
def run_prophet(df_hist, df_payday, df_holiday, target_col, start_fcst, end_fcst): 
    df_prophet = pd.DataFrame({
        'ds': df_hist['Datetime'],
        'y': df_hist[f'{target_col}_cleansed']
    })
    
    # Siapkan Holidays
    holidays_list = []
    if df_payday is not None and not df_payday.empty:
        payday_df = pd.DataFrame({'holiday': 'payday', 'ds': pd.to_datetime(df_payday['Date']), 'lower_window': 0, 'upper_window': 0})
        holidays_list.append(payday_df)
    if df_holiday is not None and not df_holiday.empty:
        nat_holiday_df = pd.DataFrame({'holiday': 'national_holiday', 'ds': pd.to_datetime(df_holiday['Date']), 'lower_window': 0, 'upper_window': 0})
        holidays_list.append(nat_holiday_df)
        
    holidays = pd.concat(holidays_list, ignore_index=True) if holidays_list else None
        
    model = Prophet(holidays=holidays, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df_prophet)
    
    # Hitung MAPE di data historis
    historical_pred = model.predict(df_prophet[['ds']])
    mape = mean_absolute_percentage_error(df_prophet['y'], historical_pred['yhat']) * 100
    
    # Buat dataframe untuk rentang tanggal forecast khusus (interval 30 menit)
    end_fcst_dt = pd.to_datetime(end_fcst) + pd.Timedelta(hours=23, minutes=30)
    future_dates = pd.date_range(start=pd.to_datetime(start_fcst), end=end_fcst_dt, freq='30T')
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Eksekusi Forecast
    forecast = model.predict(future_df)
    future_forecast = forecast[['ds', 'yhat']].copy()
    future_forecast.rename(columns={'ds': 'Datetime', 'yhat': f'{target_col}_forecast'}, inplace=True)
    future_forecast[f'{target_col}_forecast'] = future_forecast[f'{target_col}_forecast'].clip(lower=0) 
    
    return future_forecast, mape

# --- UI SIDEBAR ---
st.sidebar.header("1. Upload Database")
file_cof = st.sidebar.file_uploader("Upload Data COF (Interval 30 Min)", type=['csv', 'xlsx'])
file_aht = st.sidebar.file_uploader("Upload Data AHT (Interval 30 Min)", type=['csv', 'xlsx'])
file_payday = st.sidebar.file_uploader("Upload Data Pay Day (Opsional)", type=['csv', 'xlsx'])
file_holiday = st.sidebar.file_uploader("Upload Data Libur Nasional (Opsional)", type=['csv', 'xlsx'])

st.sidebar.header("2. Konfigurasi Tanggal")
st.sidebar.subheader("Data Historis (Training)")
start_history_date = st.sidebar.date_input("Mulai Data Historis", datetime.date(2025, 12, 1))
end_history_date = st.sidebar.date_input("Akhir Data Historis", datetime.date(2026, 2, 28))

st.sidebar.subheader("Target Forecast")
start_forecast_date = st.sidebar.date_input("Mulai Forecast", datetime.date(2026, 3, 1))
end_forecast_date = st.sidebar.date_input("Akhir Forecast", datetime.date(2026, 3, 31))

st.sidebar.header("3. Parameter Erlang C")
shrinkage = st.sidebar.number_input("Shrinkage (%)", min_value=0.0, max_value=100.0, value=30.0) / 100
max_wait_time = st.sidebar.number_input("Max Waiting Time (Detik)", value=20)
work_hours = st.sidebar.number_input("Jam Kerja per Hari", value=8)
work_days = st.sidebar.number_input("Hari Kerja per Bulan", value=22)

# Hitung jumlah hari forecast untuk kalkulasi headcount
forecast_days = (end_forecast_date - start_forecast_date).days + 1

# --- PROSES UTAMA ---
if st.button("Jalankan Forecast & Kalkulasi", type="primary"):
    if file_cof and file_aht:
        # Validasi Tanggal
        if start_history_date > end_history_date:
            st.error("Error: 'Mulai Data Historis' tidak boleh lebih besar dari 'Akhir Data Historis'.")
        elif start_forecast_date > end_forecast_date:
            st.error("Error: 'Mulai Forecast' tidak boleh lebih besar dari 'Akhir Forecast'.")
        else:
            with st.spinner("Memproses data, memfilter tanggal, dan membuat peramalan..."):
                # 1. Baca Data
                df_cof = pd.read_csv(file_cof) if file_cof.name.endswith('csv') else pd.read_excel(file_cof)
                df_aht = pd.read_csv(file_aht) if file_aht.name.endswith('csv') else pd.read_excel(file_aht)
                
                df_payday = pd.read_csv(file_payday) if (file_payday and file_payday.name.endswith('csv')) else pd.read_excel(file_payday) if file_payday else None
                df_holiday = pd.read_csv(file_holiday) if (file_holiday and file_holiday.name.endswith('csv')) else pd.read_excel(file_holiday) if file_holiday else None
                
                df_cof['Datetime'] = pd.to_datetime(df_cof['Datetime'])
                df_aht['Datetime'] = pd.to_datetime(df_aht['Datetime'])
                
                # 2. Filter Data Historis berdasarkan input tanggal
                start_hist_dt = pd.to_datetime(start_history_date)
                end_hist_dt = pd.to_datetime(end_history_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                
                df_cof = df_cof[(df_cof['Datetime'] >= start_hist_dt) & (df_cof['Datetime'] <= end_hist_dt)]
                df_aht = df_aht[(df_aht['Datetime'] >= start_hist_dt) & (df_aht['Datetime'] <= end_hist_dt)]
                
                if df_cof.empty or df_aht.empty:
                    st.error("Data kosong! Rentang 'Data Historis' yang Anda pilih tidak ditemukan di file yang diunggah.")
                else:
                    # 3. Cleansing Data
                    df_cof['COF_cleansed'], _ = cleanse_data_hw(df_cof, 'COF')
                    df_aht['AHT_cleansed'], _ = cleanse_data_hw(df_aht, 'AHT')
                    
                    # 4. Forecasting Prophet
                    forecast_cof, mape_cof = run_prophet(df_cof, df_payday, df_holiday, 'COF', start_forecast_date, end_forecast_date)
                    forecast_aht, mape_aht = run_prophet(df_aht, df_payday, df_holiday, 'AHT', start_forecast_date, end_forecast_date)
                    
                    # Gabungkan Hasil Forecast
                    df_result = pd.merge(forecast_cof, forecast_aht, on='Datetime')
                    
                    # 5. Kalkulasi Erlang C per Interval
                    df_result['Base_Agent_Needed'] = 0
                    df_result['Projected_Wait_Time'] = 0.0
                    
                    for index, row in df_result.iterrows():
                        agents, wait_time = calculate_agents_erlang(row['COF_forecast'], row['AHT_forecast'], max_wait_time)
                        df_result.at[index, 'Base_Agent_Needed'] = agents
                        df_result.at[index, 'Projected_Wait_Time'] = wait_time
                        
                    # Hitung Agent Adjusted (Dengan Shrinkage)
                    df_result['Agent_Needed_Adjust'] = np.ceil(df_result['Base_Agent_Needed'] / (1 - shrinkage))
                    
                    # Kalkulasi Headcount Bulanan
                    df_result['Date'] = df_result['Datetime'].dt.date
                    max_agents_per_day = df_result.groupby('Date')['Agent_Needed_Adjust'].max()
                    avg_daily_headcount_needed = max_agents_per_day.mean()
                    
                    total_monthly_headcount = math.ceil((avg_daily_headcount_needed * forecast_days) / work_days)

                    # --- TAMPILAN HASIL (UI) ---
                    st.success(f"Proses Selesai! Forecast dibuat untuk {forecast_days} hari.")
                    
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
                            st.warning("⚠️ Nilai MAPE di atas 15%. Anda mungkin perlu memperlebar rentang Data Historis agar model dapat belajar lebih banyak pola.")
                        else:
                            st.success("✅ Akurasi model sangat baik berdasarkan data latih yang dipilih.")
                            
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
