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
import pulp

# --- SUPPRESS WARNINGS & PROPHET LOGS ---
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').disabled = True

# --- INISIALISASI SESSION STATE (MEMORI WEB) ---
if 'agent_data' not in st.session_state:
    st.session_state['agent_data'] = pd.DataFrame()
if 'shift_target' not in st.session_state:
    st.session_state['shift_target'] = pd.DataFrame()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="WFM & Capacity Planner", layout="wide")
st.title("WFM Forecast, Capacity & Rostering System")

# --- FALLBACK KAMUS SHIFT 24 JAM (DEFAULT) ---
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

# --- FUNGSI ERLANG C ---
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

# --- FUNGSI CLEANSING & PROPHET ---
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
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Forecast & Planner", "🗄️ Database Agent & Target", "🤖 Auto Rostering"]
)
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
    
    st.subheader("👥 1. Master Data Agent")
    upload_agent = st.file_uploader("📥 Upload Excel Data Agent", type=['xlsx', 'csv'])
    
    # Inisialisasi default jika belum upload
    df_agent_display = st.session_state['agent_data'] if not st.session_state['agent_data'].empty else pd.DataFrame(columns=["Nama", "Skill", "Team Leader", "Gender", "ID Login"])

    if upload_agent:
        df_agent_display = pd.read_csv(upload_agent) if upload_agent.name.endswith('csv') else pd.read_excel(upload_agent)

    edited_agent = st.data_editor(df_agent_display, num_rows="dynamic", use_container_width=True, height=250)
    # Simpan perubahan secara realtime ke dalam Session State memori web
    st.session_state['agent_data'] = edited_agent
    
    st.divider()
    st.subheader("🧩 2. Target Komposisi Shift")
    upload_comp = st.file_uploader("📥 Upload Target Komposisi (Misal: Distribusi_Shift.xlsx)", type=['xlsx', 'csv'])
    
    df_comp_display = st.session_state['shift_target'] if not st.session_state['shift_target'].empty else pd.DataFrame(columns=["Tanggal", "S1", "S2", "Total_Agent_Shift"])

    if upload_comp:
        try:
            xls = pd.ExcelFile(upload_comp)
            df_comp_display = pd.read_excel(upload_comp, sheet_name='Distribusi_Shift') if 'Distribusi_Shift' in xls.sheet_names else pd.read_excel(upload_comp)
            if 'Tanggal' in df_comp_display.columns: df_comp_display['Tanggal'] = pd.to_datetime(df_comp_display['Tanggal']).dt.strftime('%Y-%m-%d')
        except:
            df_comp_display = pd.read_csv(upload_comp)

    edited_comp = st.data_editor(df_comp_display, num_rows="dynamic", use_container_width=True, height=250)
    # Simpan perubahan target shift ke dalam memori web
    st.session_state['shift_target'] = edited_comp

# ==========================================
# HALAMAN 3: AUTO ROSTERING MACHINE
# ==========================================
elif menu == "🤖 Auto Rostering":
    st.header("🤖 Mesin Auto-Rostering Jadwal")
    st.markdown("Halaman ini akan menjodohkan **Database Agent** dengan **Target Komposisi Shift** secara otomatis dan adil menggunakan algoritma pengacakan (Randomized Greedy Assignment).")

    agent_df = st.session_state.get('agent_data', pd.DataFrame())
    comp_df = st.session_state.get('shift_target', pd.DataFrame())

    if agent_df.empty or comp_df.empty:
        st.warning("⚠️ Data Agent atau Target Komposisi belum lengkap. Silakan lengkapi di Halaman 'Database Agent & Target' terlebih dahulu.")
    else:
        st.success(f"✅ Sistem mendeteksi **{len(agent_df)} Agen aktif** dan target jadwal untuk **{len(comp_df)} Hari**.")
        
        if st.button("🚀 Jalankan Auto Roster", type="primary", use_container_width=True):
            with st.spinner("Mengacak dan mendistribusikan shift secara adil..."):
                try:
                    name_col = next((c for c in agent_df.columns if 'nama' in c.lower()), agent_df.columns[0])
                    master_agents = agent_df[name_col].dropna().astype(str).tolist()
                    roster_records = []
                    
                    for _, row in comp_df.iterrows():
                        date_val = row.get('Tanggal', 'Unknown Date')
                        
                        # --- BLOK KODE YANG DIPERBAIKI (BUG FIX: SAFE PARSING) ---
                        shift_reqs = {}
                        for col in comp_df.columns:
                            if col.strip().lower() not in ['tanggal', 'total_agent_shift', 'date'] and pd.notna(row[col]):
                                try:
                                    # Konversi ke float dulu untuk antisipasi nilai desimal, lalu ke integer
                                    count = int(float(row[col]))
                                    if count > 0:
                                        shift_reqs[col] = count
                                except ValueError:
                                    # Jika kolom mengandung string murni (misal judul shift 'S1'), maka diabaikan
                                    continue
                        # --------------------------------------------------------
                        
                        np.random.shuffle(master_agents)
                        daily_assignment = {'Nama Agen': master_agents.copy()}
                        assigned_dict = {}
                        agent_idx = 0
                        
                        for shift_code, count in shift_reqs.items():
                            for _ in range(count):
                                if agent_idx < len(master_agents):
                                    assigned_dict[master_agents[agent_idx]] = shift_code
                                    agent_idx += 1
                                    
                        for ag in master_agents:
                            if ag not in assigned_dict:
                                assigned_dict[ag] = 'OFF'
                                
                        roster_records.append({'Tanggal': date_val, 'Assignments': assigned_dict})

                    final_roster = pd.DataFrame({'Nama Agen': master_agents})
                    for record in roster_records:
                        date_col = record['Tanggal']
                        final_roster[date_col] = final_roster['Nama Agen'].map(record['Assignments'])
                    
                    final_roster = final_roster.set_index('Nama Agen')
                    st.session_state['final_roster'] = final_roster
                    
                except Exception as e:
                    st.error(f"Gagal menjalankan Auto-Roster: {e}")

        if 'final_roster' in st.session_state:
            df_roster = st.session_state['final_roster']
            
            def style_auto_roster(val):
                if pd.isna(val) or str(val).strip() == '': return ''
                val_str = str(val).strip().upper()
                if val_str == 'OFF': return 'background-color: #ff0000; color: white; font-weight: bold; text-align: center;'
                elif val_str.startswith('S') or val_str[0].isdigit(): return 'background-color: #e6f2ff; color: #004085; text-align: center;'
                return 'text-align: center;'

            st.subheader("📋 Hasil Jadwal Roster Otomatis")
            st.dataframe(df_roster.style.map(style_auto_roster), use_container_width=True, height=500)
            
            out_excel = io.BytesIO()
            with pd.ExcelWriter(out_excel, engine='xlsxwriter') as writer:
                df_roster.to_excel(writer, sheet_name="Roster_Jadwal")
                
            st.download_button(
                label="📥 Download Jadwal Excel",
                data=out_excel.getvalue(),
                file_name="Auto_Generated_Roster.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
