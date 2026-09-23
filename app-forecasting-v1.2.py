import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import datetime
import re
import traceback
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import logging
from functools import lru_cache
import pulp

# --- SUPPRESS WARNINGS & PROPHET LOGS ---
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').disabled = True

# --- INISIALISASI SESSION STATE ---
if 'agent_data' not in st.session_state:
    st.session_state['agent_data'] = pd.DataFrame()
if 'shift_target' not in st.session_state:
    st.session_state['shift_target'] = pd.DataFrame()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="WFM & Capacity Planner", layout="wide")
st.title("WFM Forecast, Capacity & Rostering System")

# --- FALLBACK KAMUS SHIFT 24 JAM ---
DEFAULT_SHIFTS = {
    'S1': '06:00:00', 'S2': '07:00:00', 'S2.3': '07:30:00', 'S3': '08:00:00', 
    'S3.3': '08:30:00', 'S4': '09:00:00', 'S5': '10:00:00', 'S6': '11:00:00', 
    'S6.3': '11:30:00', 'S7': '12:00:00', 'S8': '13:00:00', 'S9': '14:00:00', 
    'S10': '15:00:00', 'S10.3': '15:30:00', 'S19': '19:00:00', 'S20': '20:00:00', 
    'S11': '21:00:00'
}

# --- FUNGSI ALOKASI SHIFT (PULP) ---
@st.cache_data(show_spinner=False)
def optimize_shift_distribution_pulp(df_result, master_shifts, shift_duration_hours=9.0, target_mode='Base'):
    shift_items = {}
    for s_code, s_time in master_shifts.items():
        shift_items[s_code] = pd.to_timedelta(str(s_time))
        
    results = []
    df_process = df_result.copy()
    df_process['Datetime'] = pd.to_datetime(df_process['Datetime'])
    df_process['Date'] = pd.to_datetime(df_process['Date']).dt.date
    unique_dates = df_process['Date'].unique()
    col_target = 'Base_Agent_Needed' if target_mode == 'Base' else 'Agent_Needed_Adjust'
    
    for d in unique_dates:
        df_day = df_process[df_process['Date'] == d].sort_values('Datetime').copy()
        prob = pulp.LpProblem(f"Shift_Allocation_{d}", pulp.LpMinimize)
        shift_vars = {s_code: pulp.LpVariable(f"{s_code}", lowBound=0, cat='Integer') for s_code in shift_items.keys()}
        surplus_vars = []
        
        for _, row in df_day.iterrows():
            dt = row['Datetime']
            req = row[col_target]
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
            surplus_vars.append(surplus)
            prob += pulp.lpSum(active_shifts_in_interval) - surplus == req, f"Req_{interval_str}"
            
        prob += pulp.lpSum([shift_vars[s_code] for s_code in shift_items.keys()]) + 0.01 * pulp.lpSum(surplus_vars)
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

# --- FUNGSI ERLANG & AI ---
@lru_cache(maxsize=100000)
def erlang_c_prob(agents, traffic_rounded):
    if agents <= traffic_rounded: return 1.0
    erlang_b_inv = 1.0
    for i in range(1, int(agents) + 1): erlang_b_inv = 1.0 + erlang_b_inv * i / traffic_rounded
    erlang_b = 1.0 / erlang_b_inv
    erlang_c = erlang_b / (1.0 - (traffic_rounded / agents) * (1.0 - erlang_b))
    return max(0.0, min(1.0, erlang_c))

def calculate_agents_erlang(cof, aht_seconds, target_sl, max_wait_time, interval_seconds=1800):
    if pd.isna(cof) or pd.isna(aht_seconds) or cof <= 0: return 0, 0.0, 1.0
    traffic = (cof * aht_seconds) / interval_seconds
    traffic_rounded = round(traffic, 2)
    agents = math.ceil(traffic)
    if agents == 0: agents = 1
    while True:
        prob_wait = erlang_c_prob(agents, traffic_rounded)
        if agents > traffic:
            asa = prob_wait * (aht_seconds / (agents - traffic))
            sl = 1 - (prob_wait * math.exp(-(agents - traffic) * max_wait_time / aht_seconds))
        else:
            asa = float('inf')
            sl = 0.0
        if sl >= target_sl: break
        agents += 1
    return agents, asa, sl

@st.cache_data(show_spinner=False)
def cleanse_data_hw(df, target_col, seasonal_periods=7, threshold=2.0, min_residual=15):
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

@st.cache_data(show_spinner=False)
def run_prophet_daily(df_hist_daily, df_holidays, target_col, start_fcst, end_fcst, use_auto_payday=True):
    df_prophet = pd.DataFrame({'ds': pd.to_datetime(df_hist_daily['Date']), 'y': df_hist_daily[target_col]})
    holidays_list = []
    holiday_dates = []
    
    if df_holidays is not None and not df_holidays.empty:
        if 'Tanggal' in df_holidays.columns:
            h_df = pd.DataFrame({'holiday': 'libur_nasional', 'ds': pd.to_datetime(df_holidays['Tanggal']), 'lower_window': 0, 'upper_window': 0})
            holidays_list.append(h_df)
            holiday_dates = pd.to_datetime(df_holidays['Tanggal']).dt.normalize().tolist()
            
    if use_auto_payday:
        dr = pd.date_range(start=df_prophet['ds'].min(), end=pd.to_datetime(end_fcst))
        months = dr.to_period('M').unique()
        paydays = []
        for m in months:
            paydays.append(pd.Timestamp(year=m.year, month=m.month, day=1))
            dt_25 = pd.Timestamp(year=m.year, month=m.month, day=25)
            while dt_25.weekday() >= 5 or dt_25 in holiday_dates: dt_25 -= pd.Timedelta(days=1)
            paydays.append(dt_25)
        holidays_list.append(pd.DataFrame({'holiday': 'payday', 'ds': list(set(paydays)), 'lower_window': 0, 'upper_window': 0}))
        
    final_holidays = pd.concat(holidays_list, ignore_index=True) if holidays_list else None
    try:
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
    except Exception as e:
        st.error(f"Gagal memproses AI Prophet: {e}")
        return pd.DataFrame()

# ==========================================
# UI SIDEBAR UTAMA (NAVIGASI)
# ==========================================
st.sidebar.title("🧭 Navigasi Menu")
menu = st.sidebar.radio("Pilih Halaman:", ["📊 Forecast & Planner", "🗄️ Database Agent & Target", "🤖 Auto Rostering"])
st.sidebar.divider()

# ==========================================
# HALAMAN 1: FORECAST & PLANNER
# ==========================================
if menu == "📊 Forecast & Planner":
    st.header("WFM Forecast & Capacity Planning")
    st.sidebar.header("📂 1. Upload Database")
    file_cof = st.sidebar.file_uploader("Upload Data COF", type=['csv', 'xlsx'])
    file_aht = st.sidebar.file_uploader("Upload Data AHT", type=['csv', 'xlsx'])
    file_shift = st.sidebar.file_uploader("Upload Master Shift", type=['csv', 'xlsx'])
    file_holidays = st.sidebar.file_uploader("Upload Data Libur", type=['csv', 'xlsx'])

    st.sidebar.header("⚙️ 2. Konfigurasi")
    opt_target = st.sidebar.radio("Strategi Optimasi:", ["Base (Kebutuhan Murni)", "Shrinkage (Ideal)"])
    target_sl = st.sidebar.slider("Target Service Level (%)", 50, 100, 90) / 100
    max_wait_time = st.sidebar.number_input("Target ASA (Detik)", value=20)
    max_occupancy = st.sidebar.slider("Target Max Occupancy (%)", 50, 100, 85) / 100
    shrinkage = st.sidebar.number_input("Shrinkage (%)", 0.0, 100.0, 30.0) / 100
    work_hours = st.sidebar.number_input("Jam Kerja per Hari (FTE)", 1.0, 24.0, 8.0, 0.5)
    work_days = st.sidebar.number_input("Hari Kerja/Bulan", value=22)
    shift_duration = st.sidebar.number_input("Durasi 1 Shift (Jam)", 1.0, 24.0, 9.0, 0.5)

    st.sidebar.header("📅 3. Tanggal")
    start_hist = st.sidebar.date_input("Mulai Data Historis", pd.to_datetime('2024-02-01'))
    end_hist = st.sidebar.date_input("Akhir Data Historis", pd.to_datetime('2026-05-25'))
    start_forecast = st.sidebar.date_input("Mulai Forecast", pd.to_datetime('2026-06-01'))
    end_forecast = st.sidebar.date_input("Akhir Forecast", pd.to_datetime('2026-06-30'))

    if st.button("Jalankan Forecast & Kalkulasi", type="primary"):
        if file_cof and file_aht:
            active_shifts = DEFAULT_SHIFTS.copy()
            with st.spinner("Memproses Data..."):
                def robust_wfm_parser(uploaded_file, col_target):
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
                    if df[col_target].dtype == 'object': df[col_target] = df[col_target].astype(str).str.replace(',', '.').astype(float)
                    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', dayfirst=True)
                    df = df.dropna(subset=['Datetime']) 
                    df['Datetime'] = df['Datetime'].dt.round('30min')
                    mask = (df['Datetime'] >= pd.to_datetime(start_hist)) & (df['Datetime'] <= pd.to_datetime(end_hist) + pd.Timedelta(days=1, seconds=-1))
                    df = df[mask].copy()
                    df = df.groupby('Datetime')[col_target].first().reset_index()
                    return df.set_index('Datetime').resample('30min').asfreq().fillna(0).reset_index()

                df_cof = robust_wfm_parser(file_cof, 'COF')
                df_aht = robust_wfm_parser(file_aht, 'AHT')
                
                df_cof_daily = df_cof.groupby(df_cof['Datetime'].dt.date)['COF'].sum().reset_index()
                df_cof_daily.columns = ['Date', 'COF']
                df_cof_daily['Date'] = pd.to_datetime(df_cof_daily['Date'])
                df_cof_daily['COF_cleansed'], _ = cleanse_data_hw(df_cof_daily, 'COF')
                
                df_aht_daily = df_aht.replace(0, np.nan).groupby(df_aht['Datetime'].dt.date)['AHT'].mean().reset_index()
                df_aht_daily.columns = ['Date', 'AHT']
                df_aht_daily['Date'] = pd.to_datetime(df_aht_daily['Date'])
                df_aht_daily['AHT'] = df_aht_daily['AHT'].ffill()
                df_aht_daily['AHT_cleansed'], _ = cleanse_data_hw(df_aht_daily, 'AHT')

                fcst_cof = run_prophet_daily(df_cof_daily, None, 'COF_cleansed', start_forecast, end_forecast, True)
                fcst_aht = run_prophet_daily(df_aht_daily, None, 'AHT_cleansed', start_forecast, end_forecast, True)

                profile_start = df_cof['Datetime'].max() - pd.DateOffset(months=3)
                df_recent = pd.merge(df_cof[df_cof['Datetime'] >= profile_start], df_aht[df_aht['Datetime'] >= profile_start], on='Datetime')
                df_recent['Time'] = df_recent['Datetime'].dt.time
                df_recent['Is_Weekend'] = df_recent['Datetime'].dt.weekday >= 5
                df_recent['Date_Only'] = df_recent['Datetime'].dt.date
                daily_totals = df_recent.groupby(['Date_Only', 'Is_Weekend'])['COF'].sum().reset_index(name='Tot_COF')
                df_recent = pd.merge(df_recent, daily_totals, on=['Date_Only', 'Is_Weekend'])
                df_recent['COF_Ratio'] = np.where(df_recent['Tot_COF'] > 0, df_recent['COF'] / df_recent['Tot_COF'], 0)
                profile = df_recent.groupby(['Is_Weekend', 'Time']).agg(COF_Ratio=('COF_Ratio', 'mean')).reset_index()
                profile['COF_Ratio'] = profile['COF_Ratio'] / profile.groupby('Is_Weekend')['COF_Ratio'].transform('sum')

                recon_rows = []
                for d in pd.date_range(start_forecast, end_forecast):
                    d_date = d.date()
                    is_wkd = d.weekday() >= 5
                    c_val = fcst_cof[fcst_cof['Date'] == pd.to_datetime(d_date)]['COF_cleansed_daily_forecast'].values[0] if not fcst_cof[fcst_cof['Date'] == pd.to_datetime(d_date)].empty else 0
                    a_val = fcst_aht[fcst_aht['Date'] == pd.to_datetime(d_date)]['AHT_cleansed_daily_forecast'].values[0] if not fcst_aht[fcst_aht['Date'] == pd.to_datetime(d_date)].empty else 100
                    sub_p = profile[profile['Is_Weekend'] == is_wkd]
                    for _, pr in sub_p.iterrows():
                        recon_rows.append({'Datetime': pd.Timestamp.combine(d_date, pr['Time']), 'COF_forecast': c_val * pr['COF_Ratio'], 'AHT_forecast': a_val})
                        
                df_result = pd.DataFrame(recon_rows)
                df_result['COF_forecast'] = np.ceil(df_result['COF_forecast']).astype(int)
                
                erlang_res = [calculate_agents_erlang(c, a, target_sl, max_wait_time) for c, a in zip(df_result['COF_forecast'], df_result['AHT_forecast'])]
                df_result['Base_Agent_Needed'] = [r[0] for r in erlang_res]
                df_result['Agent_Needed_Adjust'] = np.ceil(df_result['Base_Agent_Needed'] / (1 - shrinkage))
                df_result['Date'] = df_result['Datetime'].dt.date
                
                mode = 'Base' if 'Base' in opt_target else 'Adjusted'
                df_shift_dist = optimize_shift_distribution_pulp(df_result, active_shifts, shift_duration, target_mode=mode)
                
            st.success("🎉 Forecast Selesai!")
            st.dataframe(df_shift_dist, use_container_width=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_shift_dist.to_excel(writer, sheet_name='Distribusi_Shift', index=False)
            st.download_button("📥 Download Excel Forecast", output.getvalue(), "Forecast_Result.xlsx")

# ==========================================
# HALAMAN 2: DATABASE AGENT & TARGET
# ==========================================
elif menu == "🗄️ Database Agent & Target":
    st.header("🗄️ Manajemen Database Agent & Target Komposisi")
    
    st.subheader("👥 1. Master Workplace / Availability Agent")
    st.info("Pastikan memiliki kolom 'Gender' (L/P) dan 'Kondisi' (Hamil/-) di sebelah Nama Agen.")
    upload_agent = st.file_uploader("📥 Upload Excel Workplace", type=['xlsx', 'csv'])
    
    df_agent_display = st.session_state['agent_data'] if not st.session_state['agent_data'].empty else pd.DataFrame(columns=["Nama Agen", "Gender", "Kondisi", "2026-09-01", "2026-09-02"])
    
    if upload_agent:
        try:
            df_agent_display = pd.read_excel(upload_agent) if upload_agent.name.endswith('xlsx') else pd.read_csv(upload_agent)
            new_cols = []
            for col in df_agent_display.columns:
                if isinstance(col, datetime.datetime):
                    new_cols.append(col.strftime('%Y-%m-%d'))
                else:
                    new_cols.append(str(col))
            df_agent_display.columns = new_cols
        except Exception as e:
            st.error(f"Error membaca file workplace: {e}")

    edited_agent = st.data_editor(df_agent_display, num_rows="dynamic", use_container_width=True, height=250)
    st.session_state['agent_data'] = edited_agent
    
    st.divider()
    st.subheader("🧩 2. Target Komposisi Shift")
    upload_comp = st.file_uploader("📥 Upload Target Komposisi", type=['xlsx', 'csv'])
    
    df_comp_display = st.session_state['shift_target'] if not st.session_state['shift_target'].empty else pd.DataFrame(columns=["Tanggal", "S1", "S2", "Total_Agent_Shift"])

    if upload_comp:
        try:
            xls = pd.ExcelFile(upload_comp)
            df_comp_display = pd.read_excel(upload_comp, sheet_name='Distribusi_Shift') if 'Distribusi_Shift' in xls.sheet_names else pd.read_excel(upload_comp)
            if 'Tanggal' in df_comp_display.columns: df_comp_display['Tanggal'] = pd.to_datetime(df_comp_display['Tanggal']).dt.strftime('%Y-%m-%d')
        except:
            df_comp_display = pd.read_csv(upload_comp)

    edited_comp = st.data_editor(df_comp_display, num_rows="dynamic", use_container_width=True, height=250)
    st.session_state['shift_target'] = edited_comp

# ==========================================
# HALAMAN 3: AUTO ROSTERING MACHINE
# ==========================================
elif menu == "🤖 Auto Rostering":
    st.header("🤖 Mesin Auto-Rostering Berbasis Ketersediaan & Keadilan")

    st.sidebar.header("⚙️ Aturan Keadilan Jadwal")
    target_off_or = st.sidebar.number_input("Target Jumlah OFF + OR / Bulan (Bulan Target)", min_value=4, max_value=15, value=10)
    target_consecutive = st.sidebar.slider("Target Double OFF/OR berdekatan", min_value=0, max_value=4, value=2)
    
    agent_df = st.session_state.get('agent_data', pd.DataFrame())
    comp_df = st.session_state.get('shift_target', pd.DataFrame())

    if agent_df.empty or comp_df.empty:
        st.warning("⚠️ Data Workplace atau Target Komposisi belum lengkap.")
    else:
        name_col = next((c for c in agent_df.columns if 'nama' in str(c).lower()), agent_df.columns[0])
        gender_col = next((c for c in agent_df.columns if 'gender' in str(c).lower() or 'kelamin' in str(c).lower()), None)
        kondisi_col = next((c for c in agent_df.columns if 'kondisi' in str(c).lower() or 'hamil' in str(c).lower()), None)

        roster_df = agent_df.copy()
        
        date_col_name = None
        for c in comp_df.columns:
            if str(c).strip().lower() in ['tanggal', 'date', 'waktu', 'hari', 'datetime', 'tgl']:
                date_col_name = c
                break
        
        daily_targets = {}
        if date_col_name:
            for _, row in comp_df.iterrows():
                raw_date = str(row[date_col_name]).strip()
                if raw_date.lower() in ['nat', 'nan', '']: continue
                try:
                    date_obj = pd.to_datetime(raw_date.split(' ')[0])
                    date_str = date_obj.strftime('%Y-%m-%d')
                    
                    reqs = {}
                    for col in comp_df.columns:
                        if col != date_col_name and str(col).strip().lower() not in ['total_agent_shift', 'total', 'unnamed: 0']:
                            try:
                                val = int(float(row[col]))
                                if val > 0: reqs[str(col).strip()] = val
                            except: pass
                    
                    if reqs:  # Pastikan ada target yang bisa diolah
                        daily_targets[date_str] = reqs
                except: pass

        # --- DYNAMIC DATE AUTO-DETECTOR ---
        # 1. Cari semua kolom berformat tanggal di file Workplace
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        all_date_cols = [str(c).strip() for c in roster_df.columns if date_pattern.match(str(c).strip())]
        all_date_cols.sort()
        
        # 2. Tentukan Tanggal Target (Mulai Bulan Aktif) berdasarkan Target Komposisi yang masuk
        if daily_targets:
            start_target_date = min(daily_targets.keys())
        else:
            start_target_date = "2099-12-31" # Fallback safety
            
        # 3. Belah data menjadi Buffer (Histori) dan Target (Bulan Aktif)
        target_dates = [c for c in all_date_cols if c >= start_target_date]
        buffer_dates = [c for c in all_date_cols if c < start_target_date]
        
        # Buat label dinamis untuk UI
        if start_target_date != "2099-12-31":
            target_month_name = pd.to_datetime(start_target_date).strftime('%B %Y')
        else:
            target_month_name = "Bulan Berjalan"
        
        total_required_shifts = sum(sum(reqs.values()) for reqs in daily_targets.values())
        total_days = len(target_dates) # Hanya menghitung hari di bulan aktif untuk target OFF
        total_capacity = 0
        agent_initial_stats = {}
        
        for idx in roster_df.index:
            # 1. Hitung jatah kerja bulan Target murni
            leave_count_target = 0
            or_count_target = 0
            for col in target_dates:
                val = str(roster_df.at[idx, col]).strip().upper()
                if val in ['CT', 'CUTI', 'SICK', 'TRAINING']: leave_count_target += 1
                elif val == 'OR': or_count_target += 1
                
            target_work = total_days - leave_count_target - target_off_or
            total_capacity += max(0, target_work)
            
            # 2. Inisialisasi state awal dari riwayat Buffer (Initial State Reader Dinamis)
            last_shift_val = None
            consecutive_work_buffer = 0
            
            if buffer_dates:
                # Cek mundur dari tanggal terakhir di buffer ke belakang
                for buf_col in reversed(buffer_dates):
                    val = str(roster_df.at[idx, buf_col]).strip().upper()
                    if val.startswith('S') or val[0].isdigit():
                        if last_shift_val is None:
                            last_shift_val = val # Ambil shift terakhir
                        consecutive_work_buffer += 1
                    elif val in ['OFF', 'OR', 'CT', 'CUTI', 'SICK', 'TRAINING']:
                        # Ketemu hari libur di buffer, putus hitungan beruntun
                        break
            
            agent_initial_stats[idx] = {
                'target_work': target_work,
                'leave_count': leave_count_target,
                'pre_or': or_count_target,
                'initial_last_shift': last_shift_val,
                'initial_consecutive_work': min(consecutive_work_buffer, 5) # Maksimal cap di 5
            }
            
        gap = total_capacity - total_required_shifts

        st.subheader(f"🧮 Kalkulator Kapasitas vs Kebutuhan ({target_month_name})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Kebutuhan Shift", total_required_shifts)
        col2.metric("Total Kapasitas Agen", total_capacity)
        col3.metric("Selisih Keseluruhan", gap)
        st.divider()

        def is_agent_eligible(shift_code, gender, kondisi, last_shift):
            is_hamil = str(kondisi).strip().upper() == 'HAMIL'
            gender = str(gender).strip().upper()
            if gender not in ['P', 'L']: gender = 'P'

            # 1. Aturan Gender & Kondisi (Hamil)
            if is_hamil:
                if shift_code not in ['S1', 'S2', 'S2.3', 'S3']: return False
            else:
                match = re.search(r'\d+(\.\d+)?', shift_code)
                num = float(match.group()) if match else 0
                if gender == 'P' and num > 6.0: return False
                if gender == 'L' and num < 4.0: return False
                
            # 2. Aturan Anti-Jumping (Mundur Maks 1 Jam)
            if last_shift and last_shift in DEFAULT_SHIFTS and shift_code in DEFAULT_SHIFTS:
                t_last = pd.to_timedelta(DEFAULT_SHIFTS[last_shift])
                t_curr = pd.to_timedelta(DEFAULT_SHIFTS[shift_code])
                if t_curr < (t_last - pd.Timedelta(hours=1)):
                    return False
                    
            return True

        if st.button(f"🚀 Jalankan Auto Roster ({target_month_name})", type="primary", use_container_width=True):
            with st.spinner("Memproses logika antrean & keadilan shift..."):
                try:
                    # Inisialisasi Tracker Stateful dengan data awal dari Buffer (Histori)
                    agent_stats = {}
                    for idx in roster_df.index:
                        init_st = agent_initial_stats[idx]
                        agent_stats[idx] = {
                            'worked': 0, 
                            'off_or_count': init_st['pre_or'], 
                            'consecutive_count': 0, 
                            'yesterday_status': 'WORK' if init_st['initial_last_shift'] else 'OFF',
                            'consecutive_work': init_st['initial_consecutive_work'],
                            'mandatory_off_remaining': 2 if init_st['initial_consecutive_work'] >= 5 else 0,
                            'last_shift': init_st['initial_last_shift']
                        }
                    
                    sorted_dates = sorted(daily_targets.keys())

                    for target_date_str in sorted_dates:
                        shift_reqs = daily_targets.get(target_date_str, {})
                        
                        matching_col = None
                        for col in roster_df.columns:
                            if col in [name_col, gender_col, kondisi_col]: continue
                            try:
                                if pd.to_datetime(str(col)).strftime('%Y-%m-%d') == target_date_str:
                                    matching_col = col
                                    break
                            except: pass
                            
                        if matching_col:
                            is_in_mask = roster_df[matching_col].astype(str).str.strip().str.upper() == 'IN'
                            available_indices = roster_df[is_in_mask].index.tolist()

                            # 1. Saring agen yang wajib OFF karena sudah 5 HK
                            eligible_for_work_indices = []
                            for idx in available_indices:
                                if agent_stats[idx]['mandatory_off_remaining'] > 0:
                                    continue
                                eligible_for_work_indices.append(idx)

                            # 2. SCORING AGEN
                            priority_scores = []
                            for idx in eligible_for_work_indices:
                                stats = agent_stats[idx]
                                score = agent_initial_stats[idx]['target_work'] - stats['worked']
                                
                                if stats['yesterday_status'] in ['OFF', 'OR'] and stats['consecutive_count'] < target_consecutive:
                                    score -= 100 
                                
                                score += np.random.uniform(-0.5, 0.5)
                                priority_scores.append((score, idx))
                            
                            priority_scores.sort(key=lambda x: x[0], reverse=True)
                            sorted_available_indices = [x[1] for x in priority_scores]
                            
                            assigned_this_day = set()

                            # 3. ALOKASI SHIFT DENGAN LOGIKA SHIFT-FIRST & ANTI-JUMP
                            for s_code, count in shift_reqs.items():
                                for _ in range(count):
                                    for target_row in sorted_available_indices:
                                        if target_row in assigned_this_day: continue
                                        
                                        gender = roster_df.at[target_row, gender_col] if gender_col else 'P'
                                        kondisi = roster_df.at[target_row, kondisi_col] if kondisi_col else ''
                                        last_s = agent_stats[target_row]['last_shift']
                                        
                                        if is_agent_eligible(s_code, gender, kondisi, last_s):
                                            roster_df.at[target_row, matching_col] = s_code
                                            agent_stats[target_row]['worked'] += 1
                                            agent_stats[target_row]['yesterday_status'] = 'WORK'
                                            agent_stats[target_row]['last_shift'] = s_code
                                            agent_stats[target_row]['consecutive_work'] += 1
                                            agent_stats[target_row]['consecutive_count'] = 0
                                            
                                            if agent_stats[target_row]['consecutive_work'] >= 5:
                                                agent_stats[target_row]['mandatory_off_remaining'] = 2
                                                
                                            assigned_this_day.add(target_row)
                                            break

                            # 4. ALOKASI SISA AGEN (SPILLOVER KERJA ATAU OFF)
                            for target_row in available_indices:
                                if target_row not in assigned_this_day:
                                    gender = str(roster_df.at[target_row, gender_col]).strip().upper() if gender_col else 'P'
                                    kondisi = str(roster_df.at[target_row, kondisi_col]).strip().upper() if kondisi_col else ''
                                    is_hamil = (kondisi == 'HAMIL')
                                    
                                    force_work_due_to_target = (agent_stats[target_row]['off_or_count'] >= target_off_or)
                                    is_mandatory_off = (agent_stats[target_row]['mandatory_off_remaining'] > 0)
                                    
                                    if force_work_due_to_target and not is_mandatory_off:
                                        fallback_shift = None
                                        candidates = ['S3'] if is_hamil else (['S3', 'S4', 'S5', 'S6'] if gender == 'P' else ['S4', 'S5', 'S6', 'S10.3', 'S11'])
                                        for cand in candidates:
                                            if is_agent_eligible(cand, gender, kondisi, agent_stats[target_row]['last_shift']):
                                                fallback_shift = cand
                                                break
                                        if not fallback_shift:
                                            fallback_shift = agent_stats[target_row]['last_shift'] or ('S3' if gender == 'P' else 'S4')
                                            
                                        roster_df.at[target_row, matching_col] = fallback_shift
                                        agent_stats[target_row]['worked'] += 1
                                        agent_stats[target_row]['yesterday_status'] = 'WORK'
                                        agent_stats[target_row]['last_shift'] = fallback_shift
                                        agent_stats[target_row]['consecutive_work'] += 1
                                        agent_stats[target_row]['consecutive_count'] = 0
                                        if agent_stats[target_row]['consecutive_work'] >= 5:
                                            agent_stats[target_row]['mandatory_off_remaining'] = 2
                                    else:
                                        roster_df.at[target_row, matching_col] = 'OFF'
                                        if agent_stats[target_row]['yesterday_status'] in ['OFF', 'OR']:
                                            agent_stats[target_row]['consecutive_count'] += 1
                                        agent_stats[target_row]['off_or_count'] += 1
                                        agent_stats[target_row]['yesterday_status'] = 'OFF'
                                        agent_stats[target_row]['consecutive_work'] = 0
                                        agent_stats[target_row]['last_shift'] = None
                                        if agent_stats[target_row]['mandatory_off_remaining'] > 0:
                                            agent_stats[target_row]['mandatory_off_remaining'] -= 1
                                        
                                    assigned_this_day.add(target_row)
                            
                            # 5. UPDATE STATUS NON-IN (CUTI/OR)
                            not_in_mask = roster_df[matching_col].astype(str).str.strip().str.upper() != 'IN'
                            for idx in roster_df[not_in_mask].index:
                                val = str(roster_df.at[idx, matching_col]).strip().upper()
                                if val in ['OFF', 'OR']:
                                    if agent_stats[idx]['yesterday_status'] in ['OFF', 'OR']:
                                        agent_stats[idx]['consecutive_count'] += 1
                                    agent_stats[idx]['yesterday_status'] = val
                                    agent_stats[idx]['consecutive_work'] = 0
                                    agent_stats[idx]['last_shift'] = None
                                    if agent_stats[idx]['mandatory_off_remaining'] > 0:
                                        agent_stats[idx]['mandatory_off_remaining'] -= 1
                                elif val in ['CT', 'CUTI', 'SICK', 'TRAINING']:
                                    agent_stats[idx]['yesterday_status'] = 'CUTI'
                                    agent_stats[idx]['consecutive_work'] = 0
                                    agent_stats[idx]['last_shift'] = None
                                    if agent_stats[idx]['mandatory_off_remaining'] > 0:
                                        agent_stats[idx]['mandatory_off_remaining'] -= 1
                                elif val.startswith('S') or val[0].isdigit():
                                    agent_stats[idx]['yesterday_status'] = 'WORK'
                                    agent_stats[idx]['last_shift'] = val
                                    agent_stats[idx]['consecutive_work'] += 1
                                    if agent_stats[idx]['consecutive_work'] >= 5:
                                        agent_stats[idx]['mandatory_off_remaining'] = 2

                    final_roster = roster_df.set_index(name_col)
                    st.session_state['final_roster'] = final_roster
                    
                except Exception as e:
                    # Menambahkan pelacakan error yang akurat (traceback)
                    error_details = traceback.format_exc()
                    st.error(f"Terjadi kesalahan saat memproses data:\n{e}\n\nDetail Sistem:\n{error_details}")

        if 'final_roster' in st.session_state:
            df_roster = st.session_state['final_roster']
            
            def style_auto_roster(val):
                if pd.isna(val) or str(val).strip() == '': return ''
                val_str = str(val).strip().upper()
                if val_str == 'OFF': return 'background-color: #ffcccc; color: #cc0000; font-weight: bold; text-align: center;'
                elif val_str in ['CUTI', 'CT', 'OR', 'SICK', 'TRAINING']: return 'background-color: #ffe5b4; color: #cc7700; text-align: center;'
                elif val_str == 'IN': return 'background-color: #e6ffe6; color: #006600; text-align: center;'
                elif val_str.startswith('S') or val_str[0].isdigit(): return 'background-color: #e6f2ff; color: #004085; font-weight: bold; text-align: center;'
                return 'text-align: center;'

            st.subheader(f"📋 Hasil Jadwal Roster Otomatis ({target_month_name})")
            st.dataframe(df_roster.style.map(style_auto_roster), use_container_width=True, height=500)
            
            out_excel = io.BytesIO()
            with pd.ExcelWriter(out_excel, engine='xlsxwriter') as writer:
                df_roster.to_excel(writer, sheet_name=f"Roster_{target_month_name}")
                
            st.download_button(
                label="📥 Download Jadwal Excel",
                data=out_excel.getvalue(),
                file_name=f"Fairness_Based_Roster_{target_month_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
