import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Model yang sudah disimpan
model = joblib.load('diabetes_model.joblib')

# 2. Judul dan Deskripsi Web
st.title("🏥 Diabetes Risk Prediction Tool")
st.write("Aplikasi ini menggunakan Machine Learning (XGBoost) untuk mendeteksi risiko diabetes sedini mungkin.")

# 3. Sidebar untuk Input User
st.sidebar.header("Masukkan Data Pasien")

def user_input_features():
    # --- FITUR UTAMA (TOP 5) ---
    # HighBP: Tekanan Darah Tinggi
    HighBP = st.sidebar.selectbox('Tekanan Darah Tinggi (HighBP)', (0, 1), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
    
    # HighChol: Kolesterol Tinggi
    HighChol = st.sidebar.selectbox('Kolesterol Tinggi (HighChol)', (0, 1), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
    
    # BMI: Body Mass Index
    BMI = st.sidebar.slider('Body Mass Index (BMI)', 12, 98, 25)
    
    # GenHlth: Kesehatan Umum (1-5)
    GenHlth = st.sidebar.slider('Kesehatan Umum (1=Sangat Baik, 5=Sangat Buruk)', 1, 5, 3)
    
    # Age: Umur (Kategori 1-13)
    # Asumsi: 1 = 18-24 tahun, ... 13 = 80+ tahun
    Age = st.sidebar.slider('Kategori Umur (1: Muda - 13: Lansia)', 1, 13, 8)

    # --- FITUR TAMBAHAN (Diisi Default agar Error tidak muncul) ---
    # Idealnya kamu buat inputan untuk semua ini juga
    Smoker = 0
    Stroke = 0
    HeartDiseaseorAttack = 0
    PhysActivity = 1 # Asumsi aktif
    Fruits = 1
    Veggies = 1
    HvyAlcoholConsump = 0
    AnyHealthcare = 1
    NoDocbcCost = 0
    MentHlth = 0
    PhysHlth = 0
    DiffWalk = 0
    Sex = 1 # Male default
    Education = 4
    Income = 5
    CholCheck = 1 # Asumsi pernah cek kolesterol

    # Simpan dalam DataFrame sesuai urutan kolom saat training (PENTING!)
    data = {
        'HighBP': HighBP,
        'HighChol': HighChol,
        'CholCheck': CholCheck,
        'BMI': BMI,
        'Smoker': Smoker,
        'Stroke': Stroke,
        'HeartDiseaseorAttack': HeartDiseaseorAttack,
        'PhysActivity': PhysActivity,
        'Fruits': Fruits,
        'Veggies': Veggies,
        'HvyAlcoholConsump': HvyAlcoholConsump,
        'AnyHealthcare': AnyHealthcare,
        'NoDocbcCost': NoDocbcCost,
        'GenHlth': GenHlth,
        'MentHlth': MentHlth,
        'PhysHlth': PhysHlth,
        'DiffWalk': DiffWalk,
        'Sex': Sex,
        'Age': Age,
        'Education': Education,
        'Income': Income
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 4. Tampilkan Input User
st.subheader('Data Pasien:')
st.write(input_df)

# 5. Tombol Prediksi
if st.button('Prediksi Risiko Diabetes'):
    # Prediksi Kelas (0 atau 1)
    prediction = model.predict(input_df)
    # Prediksi Probabilitas (%)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader('Hasil Prediksi:')
    
    # Ambang batas (Threshold) kamu tadi 0.3
    # Kita terapkan logika threshold manual
    probabilitas_sakit = prediction_proba[0][1]
    threshold = 0.3
    
    if probabilitas_sakit >= threshold:
        st.error(f"⚠️ POSITIF DIABETES (High Risk)")
        st.error(f"⚠️ ANJAY DIABETES!")
        st.write(f"Probabilitas: {probabilitas_sakit*100:.2f}%")
        st.write("Saran: Segera konsultasi ke dokter untuk tes HbA1c (Tes Gula Darah), dan\n" \
        "STOP konsumsi makanan tinggi gula serta perbanyak aktivitas fisik.")
    else:
        st.success(f"✅ NEGATIF DIABETES (Low Risk)")
        st.success(f"✅ ANDA SEHAT!")
        st.write(f"Probabilitas: {probabilitas_sakit*100:.2f}%")
        st.write("Saran: Tetap jaga pola hidup sehat.")