# forecast
melakukan forecasting
# WFM Forecast & Capacity Planning Tool 🚀

Sistem Pendukung Keputusan (DSS) berbasis AI untuk operasional Contact Center. Aplikasi ini menggabungkan Forecasting (Prophet & Holt-Winters), Kalkulasi Kapasitas (Erlang C), dan Optimasi Jadwal Shift (PuLP Linear Programming).

## 🌟 Fitur Utama
* **Robust Parser:** Kebal terhadap isu *ghost milliseconds* dan perbedaan *locale* lintas OS (Windows/Android).
* **AI WFM Forecasting:** Prediksi volume interaksi (COF) dan durasi penanganan (AHT).
* **Smart Capacity Planning:** Kalkulasi agen berdasarkan target *Service Level* sekaligus membatasi *Occupancy Rate* agar agen tidak *burnout*.
* **Automated Shift Optimization:** Distribusi shift paling efisien menggunakan algoritma *Linear Programming*.
* **Gemini AI Insights:** Ringkasan naratif otomatis untuk Executive Summary.

## 🛠️ Cara Menjalankan Secara Lokal
1. Clone repositori ini.
2. Instal library yang dibutuhkan: `pip install -r requirements.txt`
3. Jalankan aplikasi: `streamlit run app.py`
