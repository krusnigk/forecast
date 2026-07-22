import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import datetime
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import logging
from functools import lru_cache
import pulp  # Library untuk Linear Programming

# --- SUPPRESS WARNINGS & PROPHET LOGS ---
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').disabled = True

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="WFM Forecast & Capacity Planner", layout="wide")
st.title("WFM Forecast & Capacity Planning Tool")

# --- FALLBACK KAMUS SHIFT 24 JAM (DEFAULT) ---
DEFAULT_SHIFTS = {
    'S1': '06:00:00', 'S2': '07:00:00', 'S2.3': '07:30:00', 'S3': '08:00:00', 
    'S3.3': '08:30:00', 'S4': '09:00:00', 'S5': '10:00:00', 'S6': '11:00:00', 
    'S6.3': '11:30:00', 'S7': '12:00:00', 'S8': '13:00:00', 'S9': '14:00:00', 
    'S10': '15:00:00', 'S10.3': '15:30:00', 'S19': '19:00:00', 'S20': '20:00:00', 
    'S11': '21:00:00'
}

# --- FUNGSI ALOKASI SHIFT (PULP) - STRATEGI: SCHEDULING TO BASE WITH SOFT CONSTRAINTS ---
@st.cache_data(show_spinner=False)
def optimize_shift_distribution_pulp(df_result, master_shifts, shift_duration_hours=9):
    shift_items = {}
    for s_code, s_time in master_shifts.items():
        shift_items[s_code] = pd.to_timedelta(str(s_time))
        
    results = []
    
    df_process = df_result.copy()
    df_process['Datetime'] = pd.to_datetime(df_process['Datetime'])
    df_process['Date'] = pd.to_datetime(df_process['Date']).dt.date
    
    unique_dates = df_process['Date'].unique()
    
    for d in unique_dates:
        df_day = df_process[df_process['Date'] == d].sort_values('Datetime').copy()
        
        prob = pulp.LpProblem(f"Shift_Allocation_{d}", pulp.LpMinimize)
        
        shift_vars = {s_code: pulp.LpVariable(f"{s_code}", lowBound=0, cat='Integer') for s_code in shift_items.keys()}
        
        surplus_vars = []
        shortage_vars = [] # Variabel untuk mengizinkan SL Drop (Kekurangan Agent)
        
        for _, row in df_day.iterrows():
            dt = row['Datetime']
            
            # 1. PERUBAHAN KRUSIAL: Target kini menggunakan Kebutuhan Murni Tanpa Shrinkage
            req = row['Base_Agent_Needed'] 
            
            active_shifts_in_interval = []
            
            for s_code, s_start_td in shift_items.items():
                shift_start_dt = pd.to_datetime(str(d)) + s_start_td
                shift_end_dt = shift_start_dt + pd.Timedelta(hours=shift_duration_hours)
                
                if shift_start_dt <= dt < shift_end_dt:
                    active_shifts_in_interval.append(shift_vars[s_code])
                elif (shift_start_dt - pd.Timedelta(days=1)) <= dt < (shift_end_dt - pd.Timedelta(days=1)):
                    active_shifts_in_interval.append(shift_vars[s_code])
            
            interval_str = dt.strftime('%H%M')
            
            surplus = pulp.LpVariable(f"Surplus_{interval_str}", lowBound=0)
            shortage = pulp.LpVariable(f"Shortage_{interval_str}", lowBound=0)
            
            surplus_vars.append(surplus)
            shortage_vars.append(shortage)
            
            # 2. SOFT CONSTRAINT: Agent Duty + Kekurangan - Kelebihan = Target Murni
            # Algoritma diizinkan untuk "kekurangan" asal membayar penalti di fungsi objektif
            prob += pulp.lpSum(active_shifts_in_interval) + shortage - surplus == req, f"Req_{interval_str}"
            
        # 3. FUNGSI OBJEKTIF DINAMIS:
        # - Biaya 1.0 = Untuk menambah 1 agent (shift penuh)
        # - Biaya 0.15 = Penalti per interval jika understaff (membiarkan SL merah/shortage)
        # - Biaya 0.01 = Penalti per interval jika overstaff (memaksa jadwal tetap mepet/smooth)
        prob += pulp.lpSum([shift_vars[s_code] for s_code in shift_items.keys()]) + \
                0.15 * pulp.lpSum(shortage_vars) + \
                0.01 * pulp.lpSum(surplus_vars)
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        row_res = {'Tanggal': d}
        total = 0
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            for s_code in shift_items.keys():
                val = int(shift_vars[s_code].varValue) if shift_vars[s_code].varValue is not None else 0
                row_res[s_code] = val
                total += val
        else:
            for s_code in shift_items.keys():
                row_res[s_code] = 0
                
        row_res['Total_Agent_Shift'] = total
        results.append(row_res)
        
    return pd.DataFrame(results)

# --- FUNGSI ERLANG C ITERATIF ---
@lru_cache(maxsize=100000)
def erlang_c_prob(agents, traffic_rounded):
    if agents <= traffic_rounded:
        return 1.0
    erlang_b_inv = 1.0
    for i in range(1, int(agents) + 1):
        erlang_b_inv = 1.0 + erlang_b_inv * i / traffic_rounded
    erlang_b = 1.0 / erlang_b_inv
    erlang_c = erlang_b / (1.0 - (traffic_rounded / agents) * (1.0 - erlang_b))
    return max(0.0, min(1.0, erlang_c))

def calculate_agents_erlang(cof, aht_seconds, target_sl, max_wait_time, interval_seconds=1800):
    if pd.isna(cof) or pd.isna(aht_seconds) or cof <= 0:
        return 0, 0.0, 1.0
    
    traffic = (cof * aht_seconds) / interval_seconds
    traffic_rounded = round(traffic, 2)
    
    agents = math.ceil(traffic)
    if agents == 0:
        agents = 1
    
    while True:
        prob_wait = erlang_c_prob(agents, traffic_rounded)
        if agents > traffic:
            asa = prob_wait * (aht_seconds / (agents - traffic))
            sl = 1 - (prob_wait * math.exp(-(agents - traffic) * max_wait_time / aht_seconds))
        else:
            asa = float('inf')
            sl = 0.0
            
        if sl >= target_sl:
            break
        agents += 1
        
    return agents, asa, sl

# --- FUNGSI CLEANSING HOLT-WINTERS ---
@st.cache_data(show_spinner=False)
def cleanse_data_hw(df, target_col, seasonal_periods=48, threshold=2.0, min_residual=15):
    series = df[target_col].ffill().bfill()
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated")
    hw_fit = model.fit()
    fitted_values = hw_fit.fittedvalues
    residuals = np.abs(series - fitted_values)
    std_dev = np.std(residuals)
    is_anomaly = (residuals > (threshold * std_dev)) & (residuals > min_residual)
    cleansed_series = series.copy()
    cleansed_series[is_anomaly] = fitted_values[is_anomaly]
    return cleansed_series, is_anomaly

# --- FUNGSI PROPHET UNTUK HARIAN (MACRO FORECAST) ---
@st.cache_data(show_spinner=False)
def run_prophet_daily(df_hist_daily, df_holidays, target_col, start_fcst, end_fcst, use_auto_payday=True):
    df_prophet = pd.DataFrame({
        'ds': df_hist_daily['Date'],
        'y': df_hist_daily[target_col]
    })
    
    holidays_list = []
    holiday_dates = []
    
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
            
    if use_auto_payday:
        start_date = df_prophet['ds'].min()
        end_date = pd.to_datetime(end_fcst)
        dr = pd.date_range(start=start_date, end=end_date)
        months = dr.to_period('M').unique()
        
        paydays = []
        for m in months:
            dt_1 = pd.Timestamp(year=m.year, month=m.month, day=1)
            paydays.append(dt_1)
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
        
    model = Prophet(holidays=final_holidays, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='multiplicative')
    model.fit(df_prophet)
    
    last_hist_date = df_prophet['ds'].max()
    end_fcst_dt = pd.to_datetime(end_fcst)
    
    periods = (end_fcst_dt - last_hist_date).days if end_fcst_dt > last_hist_date else 0
        
    future = model.make_future_dataframe(periods=periods, freq='D', include_history=False)
    forecast = model.predict(future)
    
    start_fcst_dt = pd.to_datetime(start_fcst)
    future_forecast = forecast[(forecast['ds'] >= start_fcst_dt) & (forecast['ds'] <= end_fcst_dt)][['ds', 'yhat']]
    future_forecast.rename(columns={'ds': 'Date', 'yhat': f'{target_col}_daily_forecast'}, inplace=True)
    future_forecast[f'{target_col}_daily_forecast'] = future_forecast[f'{target_col}_daily_forecast'].clip(lower=1)
    
    return future_forecast

# --- FUNGSI PROPHET INTERVAL UNTUK AHT ---
@st.cache_data(show_spinner=False)
def run_prophet_interval(df_hist, df_holidays, target_col, start_fcst, end_fcst, use_auto_payday=True):
    df_prophet = pd.DataFrame({
        'ds': df_hist['Datetime'],
        'y': df_hist[f'{target_col}_cleansed']
    })
    
    df_prophet['y'] = df_prophet['y'].replace(0, 1.0)
    
    holidays_list = []
    holiday_dates = []
    
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
            
    if use_auto_payday:
        start_date = df_prophet['ds'].min()
        end_date = pd.to_datetime(end_fcst)
        dr = pd.date_range(start=start_date, end=end_date)
        months = dr.to_period('M').unique()
        
        paydays = []
        for m in months:
            dt_1 = pd.Timestamp(year=m.year, month=m.month, day=1)
            paydays.append(dt_1)
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
    
    periods = int((end_fcst_dt - last_hist_date).total_seconds() / 1800) if end_fcst_dt > last_hist_date else 0
        
    future = model.make_future_dataframe(periods=periods, freq='30min', include_history=False)
    forecast = model.predict(future)
    
    start_fcst_dt = pd.to_datetime(start_fcst)
    future_forecast = forecast[(forecast['ds'] >= start_fcst_dt) & (forecast['ds'] <= end_fcst_dt)][['ds', 'yhat']]
    future_forecast.rename(columns={'ds': 'Datetime', 'yhat': f'{target_col}_forecast'}, inplace=True)
    future_forecast[f'{target_col}_forecast'] = future_forecast[f'{target_col}_forecast'].clip(lower=1)
    
    return future_forecast, 0.0

# --- UI SIDEBAR ---
st.sidebar.header("📂 1. Upload Database")
file_cof = st.sidebar.file_uploader("Upload Data COF (Interval 30 Min)", type=['csv', 'xlsx'])
file_aht = st.sidebar.file_uploader("Upload Data AHT (Interval 30 Min)", type=['csv', 'xlsx'])
file_shift = st.sidebar.file_uploader("Upload Master Shift (Opsional)", type=['csv', 'xlsx'])
file_holidays = st.sidebar.file_uploader("Upload Data Libur Nasional (Opsional)", type=['csv', 'xlsx'])

st.sidebar.header("⚙️ 2. Konfigurasi WFM")
target_sl = st.sidebar.slider("Target Service Level (%)", min_value=50, max_value=100, value=90) / 100
max_wait_time = st.sidebar.number_input("Target ASA (Detik)", value=20)
shrinkage = st.sidebar.number_input("Shrinkage (%)", min_value=0.0, max_value=100.0, value=30.0) / 100
work_hours = st.sidebar.number_input("Jam Kerja per Hari (FTE)", value=8)
work_days = st.sidebar.number_input("Hari Kerja/Agen/Bulan", value=22)
shift_duration = st.sidebar.number_input("Durasi 1 Shift (Jam)", value=9)

st.sidebar.header("📊 3. Profil Intraday")
months_profile = st.sidebar.slider("Gunakan Profil Interval (Bulan Terakhir)", min_value=1, max_value=12, value=3)

st.sidebar.header("📅 4. Konfigurasi Tanggal")
start_hist = st.sidebar.date_input("Mulai Data Historis", pd.to_datetime('2024-02-01'))
end_hist = st.sidebar.date_input("Akhir Data Historis", pd.to_datetime('2026-05-25'))
start_forecast = st.sidebar.date_input("Mulai Forecast", pd.to_datetime('2026-06-01'))
end_forecast = st.sidebar.date_input("Akhir Forecast", pd.to_datetime('2026-06-30'))

use_payday = st.sidebar.checkbox("💰 Aktifkan Auto-Payday (Tgl 1 & 25)", value=True)

# --- PROSES UTAMA ---
if st.button("Jalankan Forecast & Kalkulasi", type="primary"):
    if file_cof and file_aht:
        
        active_shifts = DEFAULT_SHIFTS.copy()
        if file_shift is not None:
            try:
                df_s = pd.read_csv(file_shift) if file_shift.name.endswith('csv') else pd.read_excel(file_shift)
                if 'Kode_Shift' in df_s.columns and 'Waktu_Mulai' in df_s.columns:
                    df_s['Waktu_Mulai'] = df_s['Waktu_Mulai'].astype(str)
                    active_shifts = dict(zip(df_s['Kode_Shift'], df_s['Waktu_Mulai']))
                    st.toast("✅ Master Shift kustom berhasil dimuat!", icon="🧩")
            except Exception as e:
                st.warning(f"Gagal membaca file Shift. Error: {e}")
                
        with st.spinner("Memvalidasi, Membaca, & Pre-processing Data Historis..."):
            df_cof = pd.read_csv(file_cof) if file_cof.name.endswith('csv') else pd.read_excel(file_cof)
            df_aht = pd.read_csv(file_aht) if file_aht.name.endswith('csv') else pd.read_excel(file_aht)
            
            df_cof['Datetime'] = pd.to_datetime(df_cof['Datetime'])
            df_aht['Datetime'] = pd.to_datetime(df_aht['Datetime'])
            
            df_cof = df_cof[(df_cof['Datetime'] >= pd.to_datetime(start_hist)) & (df_cof['Datetime'] <= pd.to_datetime(end_hist) + pd.Timedelta(days=1, seconds=-1))].copy()
            df_aht = df_aht[(df_aht['Datetime'] >= pd.to_datetime(start_hist)) & (df_aht['Datetime'] <= pd.to_datetime(end_hist) + pd.Timedelta(days=1, seconds=-1))].copy()
            
            df_cof = df_cof.set_index('Datetime').resample('30min').asfreq().fillna(0).reset_index()
            df_aht = df_aht.set_index('Datetime').resample('30min').asfreq().fillna(0).reset_index()

        with st.spinner("Melatih Model AI & Menerapkan Intraday Profiling..."):
            df_cof['COF_cleansed'], _ = cleanse_data_hw(df_cof, 'COF', min_residual=15)
            df_aht['AHT_cleansed'], _ = cleanse_data_hw(df_aht, 'AHT', min_residual=50)
            
            df_cof_daily = df_cof.groupby(df_cof['Datetime'].dt.date)['COF_cleansed'].sum().reset_index()
            df_cof_daily.columns = ['Date', 'COF']
            df_cof_daily['Date'] = pd.to_datetime(df_cof_daily['Date'])
            
            forecast_cof_daily = run_prophet_daily(df_cof_daily, df_holidays, 'COF', start_forecast, end_forecast, use_auto_payday=use_payday)
            
            max_hist_date = df_cof['Datetime'].max()
            profile_start_date = max_hist_date - pd.DateOffset(months=months_profile)
            df_recent = df_cof[df_cof['Datetime'] >= profile_start_date].copy()
            
            df_recent['Time'] = df_recent['Datetime'].dt.time
            df_recent['Is_Weekend'] = df_recent['Datetime'].dt.weekday >= 5
            df_recent['Date_Only'] = df_recent['Datetime'].dt.date
            
            daily_totals = df_recent.groupby(['Date_Only', 'Is_Weekend'])['COF_cleansed'].sum().reset_index()
            daily_totals.rename(columns={'COF_cleansed': 'Daily_Total'}, inplace=True)
            
            df_recent = pd.merge(df_recent, daily_totals, on=['Date_Only', 'Is_Weekend'])
            df_recent['Ratio'] = df_recent['COF_cleansed'] / df_recent['Daily_Total']
            
            profile = df_recent.groupby(['Is_Weekend', 'Time'])['Ratio'].mean().reset_index()
            sum_ratios = profile.groupby('Is_Weekend')['Ratio'].transform('sum')
            profile['Ratio'] = profile['Ratio'] / sum_ratios
            
            forecast_dates = pd.date_range(start=start_forecast, end=end_forecast)
            reconstructed_rows = []
            
            for d in forecast_dates:
                d_date = d.date()
                match_row = forecast_cof_daily[forecast_cof_daily['Date'] == pd.to_datetime(d_date)]
                if match_row.empty: continue
                daily_val = match_row['COF_daily_forecast'].values[0]
                is_wkd = d.weekday() >= 5
                
                sub_profile = profile[profile['Is_Weekend'] == is_wkd]
                for _, p_row in sub_profile.iterrows():
                    t = p_row['Time']
                    ratio = p_row['Ratio']
                    dt_interval = pd.Timestamp.combine(d_date, t)
                    reconstructed_rows.append({'Datetime': dt_interval, 'COF_forecast': daily_val * ratio})
                    
            forecast_cof_final = pd.DataFrame(reconstructed_rows)
            forecast_aht_final, _ = run_prophet_interval(df_aht, df_holidays, 'AHT', start_forecast, end_forecast, use_auto_payday=use_payday)
            
            df_result = pd.merge(forecast_cof_final, forecast_aht_final, on='Datetime')
            df_result['COF_forecast'] = np.ceil(df_result['COF_forecast']).astype(int)
            
        with st.spinner("Kalkulasi Antrean Erlang C..."):
            df_result['Base_Agent_Needed'] = 0
            df_result['Projected_Wait_Time'] = 0.0
            df_result['Service_Level_Achieved'] = 0.0
            
            for index, row in df_result.iterrows():
                agents, wait_time, sl_achieved = calculate_agents_erlang(row['COF_forecast'], row['AHT_forecast'], target_sl, max_wait_time)
                df_result.at[index, 'Base_Agent_Needed'] = agents
                df_result.at[index, 'Projected_Wait_Time'] = wait_time
                df_result.at[index, 'Service_Level_Achieved'] = sl_achieved
                
            df_result['Agent_Needed_Adjust'] = np.ceil(df_result['Base_Agent_Needed'] / (1 - shrinkage))
            df_result['Date'] = df_result['Datetime'].dt.date
            
            total_cof_bulan = df_result['COF_forecast'].sum()
            avg_aht_bulan = df_result['AHT_forecast'].mean()
            avg_sl_bulan = df_result['Service_Level_Achieved'].mean()
            
            kebutuhan_ws_bulan = df_result['Agent_Needed_Adjust'].max() 
            df_daily_workload_hours = df_result.groupby('Date')['Agent_Needed_Adjust'].sum() * 0.5
            daily_headcount_needed = np.ceil(df_daily_workload_hours / work_hours)
            
            total_monthly_headcount = math.ceil((daily_headcount_needed.mean() * ((pd.to_datetime(end_forecast) - pd.to_datetime(start_forecast)).days + 1)) / work_days)

            df_daily = df_result.groupby('Date').agg(
                Total_COF=('COF_forecast', 'sum'),
                Rata_Rata_AHT=('AHT_forecast', 'mean'),
                Max_Kebutuhan_Agent=('Agent_Needed_Adjust', 'max'),
                Rata_Rata_SL=('Service_Level_Achieved', 'mean')
            ).reset_index()

            df_daily['Headcount_Harian_FTE'] = df_daily['Date'].map(daily_headcount_needed)

            df_daily_display = df_daily.copy()
            df_daily_display['Total_COF'] = df_daily_display['Total_COF'].astype(int) 
            df_daily_display['Rata_Rata_AHT'] = df_daily_display['Rata_Rata_AHT'].apply(lambda x: f"{x:.0f} s")
            df_daily_display['Headcount_Harian_FTE'] = df_daily_display['Headcount_Harian_FTE'].astype(int) 
            df_daily_display['Max_Kebutuhan_Agent'] = df_daily_display['Max_Kebutuhan_Agent'].astype(int)
            df_daily_display['Rata_Rata_SL'] = df_daily_display['Rata_Rata_SL'].apply(lambda x: f"{x:.2%}")
            
            df_daily_display = df_daily_display[['Date', 'Total_COF', 'Rata_Rata_AHT', 'Headcount_Harian_FTE', 'Max_Kebutuhan_Agent', 'Rata_Rata_SL']]
            df_daily_display.columns = ['Tanggal', 'Total COF', 'Rata-rata AHT', 'Headcount Harian (FTE)', 'Kebutuhan Agent (Max/Peak)', 'Proyeksi SL']

        with st.spinner("Menjalankan Optimasi Shift LP (Base Need & Toleransi SL Drop)..."):
            df_shift_dist = optimize_shift_distribution_pulp(df_result, active_shifts, shift_duration)

        st.success("🎉 Kalkulasi Optimasi Selesai!")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Forecast", "👥 Kapasitas (Bulanan)", "📅 Hasil Harian", "⏱️ Detail Interval", "🧩 Distribusi Shift"
        ])
        
        with tab1:
            st.line_chart(df_result.set_index('Datetime')['COF_forecast'])
        with tab2:
            st.metric("Total Headcount Manusia (FTE)", total_monthly_headcount)
        with tab3:
            st.dataframe(df_daily_display, use_container_width=True)
        with tab4:
            st.dataframe(df_result[['Datetime', 'Base_Agent_Needed', 'Agent_Needed_Adjust']], use_container_width=True)
        with tab5:
            st.markdown("Distribusi Shift ini **dioptimasi berdasarkan Kebutuhan Murni** dan mengizinkan sedikit *understaffing* di jam tertentu untuk menghemat total headcount (Mirip dengan kondisi 75 Agent).")
            if not df_shift_dist.empty:
                st.dataframe(df_shift_dist, use_container_width=True)
                df_chart = df_shift_dist.set_index('Tanggal').drop(columns=['Total_Agent_Shift'])
                st.bar_chart(df_chart)

    else:
        st.info("Upload file COF dan AHT untuk memulai.")
