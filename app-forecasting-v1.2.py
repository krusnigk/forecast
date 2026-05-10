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

# --- FUNGSI ERLANG C & SERVICE LEVEL ---
def erlang_c_calculations(agents, traffic, aht, target_time):
    if agents <= traffic:
        return 1.0, float('inf'), 0.0 # Prob antrean 100%, ASA inf, SL 0%
    
    # Hitung Erlang B Iteratif
    erlang_b_inv = 1.0
    for i in range(1, int(agents) + 1):
        erlang_b_inv = 1.0 + erlang_b_inv * i / traffic
    erlang_b = 1.0 / erlang_b_inv
    
    # Hitung Probabilitas Menunggu (Erlang C)
    pw = erlang_b / (1.0 - (traffic / agents) * (1.0 - erlang_b))
    pw = max(0.0, min(1.0, pw))
    
    # Hitung ASA
    asa = pw * (aht / (agents - traffic))
    
    # Hitung Service Level (SL)
    # Formula: SL = 1 - (PW * e^(-(agents - traffic) * (target_time / aht)))
    sl = 1 - (pw * math.exp(-(agents - traffic) * (target_time / aht)))
    sl = max(0.0, min(1.0, sl))
    
    return pw, asa, sl

def find_required_agents(cof, aht_seconds, target_sl, target_time):
    if pd.isna(cof) or pd.isna(aht_seconds) or cof <= 0:
        return 0, 0.0, 0.0
    
    interval_seconds = 1800 
    traffic = (cof * aht_seconds) / interval_seconds
    agents = math.ceil(traffic) + 1
    
    while True:
        _, asa, sl = erlang_c_calculations(agents, traffic, aht_seconds, target_time)
        # Cari agen sampai memenuhi target SL (karena SL biasanya lebih ketat dibanding ASA)
        if sl >= target_sl:
            break
        agents += 1
        if agents > 1000: # Safety break
            break
            
    return agents, asa, sl

# --- FUNGSI CLEANSING HOLT-WINTERS (DENGAN PENGAMAN) ---
def cleanse_data_hw(df, target_col, seasonal_periods=48, threshold=2.0):
    series = df[target_col].ffill().bfill()
    if len(series) < (2 * seasonal_periods):
        return series, pd.Series([False] * len(series), index=series.index)
    
    try:
        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated")
        hw_fit = model.fit()
        fitted_values = hw_fit.fittedvalues
        residuals = np.abs(series - fitted_values)
        std_dev = np.std(residuals)
        if std_dev == 0: return series, pd.Series([False] * len(series), index=series.index)
        is_anomaly = residuals > (threshold * std_dev)
        cleansed_series = series.copy()
        cleansed_series[is_anomaly] = fitted_values[is_anomaly]
        return cleansed_series, is_anomaly
    except:
        return series, pd.Series([False] * len(series), index=series.index)

# --- FUNGSI FORECAST PROPHET ---
def run_prophet(df_hist, df_payday, df_holiday, target_col, start_fcst, end_fcst): 
    df_prophet = pd.DataFrame({'ds': df_hist['Datetime'], 'y': df_hist[f'{target_col}_cleansed']})
    holidays_list = []
    if df_payday is not None and not df_payday.empty:
        holidays_list.append(pd.DataFrame({'holiday': 'payday', 'ds': pd.to_datetime(df_payday['Date']), 'lower_window': 0, 'upper_window': 0}))
    if df_holiday is not None and not df_holiday.empty:
        holidays_list.append(pd.DataFrame({'holiday': 'national_holiday', 'ds': pd.to_datetime(df_holiday['Date']), 'lower_window': 0, 'upper_window': 0}))
    holidays = pd.concat(holidays_list, ignore_index=True) if holidays_list else None
    
    model = Prophet(holidays=holidays, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df_prophet)
    
    historical_pred = model.predict(df_prophet[['ds']])
    mape = mean_absolute_percentage_error(df_prophet['y'], historical_pred['yhat']) * 100
    
    end_fcst_dt = pd.to_datetime(end_fcst) + pd.Timedelta(hours=23, minutes=30)
    future_dates = pd.date_range(start=pd.to_datetime(start_fcst), end=end_fcst_dt, freq='30T')
    forecast = model.predict(pd.DataFrame({'ds': future_dates}))
    
    future_forecast = forecast[['ds', 'yhat']].copy()
    future_forecast.rename(columns={'ds': 'Datetime', 'yhat': f'{target_col}_forecast'}, inplace=True)
    future_forecast[f'{target_col}_forecast'] = future_forecast[f'{target_col}_forecast'].clip(lower=0) 
    return future_forecast, mape

# --- UI SIDEBAR ---
st.sidebar.header("1. Upload Database")
file_cof = st.sidebar.file_uploader("Upload Data COF", type=['csv', 'xlsx'])
file_aht = st.sidebar.file_uploader("Upload Data AHT", type=['csv', 'xlsx'])
file_payday = st.sidebar.file_uploader("Upload Pay Day", type=['csv', 'xlsx'])
file_holiday = st.sidebar.file_uploader("Upload Libur Nasional", type=['csv', 'xlsx'])

st.sidebar.header("2. Konfigurasi Tanggal")
start_history_date = st.sidebar.date_input("Mulai Data Historis", datetime.date(2025, 12, 1))
end_history_date = st.sidebar.date_input("Akhir Data Historis", datetime.date(2026, 2, 28))
start_forecast_date = st.sidebar.date_input("Mulai Forecast", datetime.date(2026, 3, 1))
end_forecast_date = st.sidebar.date_input("Akhir Forecast", datetime.date(2026, 3, 31))

st.sidebar.header("3. Target & Parameter WFM")
target_sl_pct = st.sidebar.slider("Target Service Level (%)", 50, 100, 80) / 100
target_asa_sec = st.sidebar.number_input("Target ASA / Threshold (Detik)", value=20)
shrinkage = st.sidebar.number_input("Shrinkage (%)", value=30.0) / 100
work_hours = st.sidebar.number_input("Jam Kerja per Hari", value=8)
work_days = st.sidebar.number_input("Hari Kerja per Bulan", value=22)

# --- PROSES UTAMA ---
if st.button("Jalankan Forecast & Kalkulasi", type="primary"):
    if file_cof and file_aht:
        with st.spinner("Mengkalkulasi..."):
            df_cof = pd.read_csv(file_cof) if file_cof.name.endswith('csv') else pd.read_excel(file_cof)
            df_aht = pd.read_csv(file_aht) if file_aht.name.endswith('csv') else pd.read_excel(file_aht)
            df_payday = pd.read_csv(file_payday) if (file_payday and file_payday.name.endswith('csv')) else pd.read_excel(file_payday) if file_payday else None
            df_holiday = pd.read_csv(file_holiday) if (file_holiday and file_holiday.name.endswith('csv')) else pd.read_excel(file_holiday) if file_holiday else None
            
            df_cof['Datetime'] = pd.to_datetime(df_cof['Datetime'])
            df_aht['Datetime'] = pd.to_datetime(df_aht['Datetime'])
            
            start_hist_dt = pd.to_datetime(start_history_date)
            end_hist_dt = pd.to_datetime(end_history_date) + pd.Timedelta(hours=23, minutes=59)
            
            df_cof = df_cof[(df_cof['Datetime'] >= start_hist_dt) & (df_cof['Datetime'] <= end_hist_dt)]
            df_aht = df_aht[(df_aht['Datetime'] >= start_hist_dt) & (df_aht['Datetime'] <= end_hist_dt)]
            
            if not df_cof.empty:
                df_cof['COF_cleansed'], _ = cleanse_data_hw(df_cof, 'COF')
                df_aht['AHT_cleansed'], _ = cleanse_data_hw(df_aht, 'AHT')
                
                fcst_cof, mape_cof = run_prophet(df_cof, df_payday, df_holiday, 'COF', start_forecast_date, end_forecast_date)
                fcst_aht, mape_aht = run_prophet(df_aht, df_payday, df_holiday, 'AHT', start_forecast_date, end_forecast_date)
                
                df_res = pd.merge(fcst_cof, fcst_aht, on='Datetime')
                
                # Kalkulasi Erlang C
                df_res['Base_Agent_Needed'] = 0
                df_res['Projected_SL'] = 0.0
                
                for i, row in df_res.iterrows():
                    agents, _, sl = find_required_agents(row['COF_forecast'], row['AHT_forecast'], target_sl_pct, target_asa_sec)
                    df_res.at[i, 'Base_Agent_Needed'] = agents
                    df_res.at[i, 'Projected_SL'] = sl
                
                df_res['Agent_Needed_Adjust'] = np.ceil(df_res['Base_Agent_Needed'] / (1 - shrinkage))
                
                # --- FORMAT OUTPUT SESUAI PERMINTAAN ---
                df_final = pd.DataFrame({
                    'Date': df_res['Datetime'].dt.date,
                    'Interval': df_res['Datetime'].dt.strftime('%H:%M'),
                    'AHT_Forecast': df_res['AHT_forecast'].round(2),
                    'Base_Agent_Erlang': df_res['Base_Agent_Needed'],
                    'Agent_With_Shrinkage': df_res['Agent_Needed_Adjust'],
                    'Projected_SL_Pct': (df_res['Projected_SL'] * 100).round(2)
                })

                st.success("Kalkulasi Selesai!")
                st.subheader("Hasil Akhir Perhitungan Kapasitas")
                st.dataframe(df_final, use_container_width=True)
                
                # Download Button
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button("Download Data (.csv)", csv, "wfm_capacity_plan.csv", "text/csv")
    else:
        st.error("Upload data terlebih dahulu.")
