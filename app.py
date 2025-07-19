import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan encoder
model = joblib.load('model_stunting_rf.pkl')

# Label encoder manual (pastikan sama dengan saat training)
gender_mapping = {'Female': 0, 'Male': 1}
status_mapping = {0: 'Normal', 1: 'Stunting', 2: 'Severely Stunting'}

# Judul Aplikasi
st.title("📊 Deteksi Stunting pada Balita")
st.write("Masukkan data anak di bawah ini untuk memprediksi status stunting:")

# Form input pengguna
with st.form(key='form_prediksi'):
    usia = st.number_input("Usia (bulan)", min_value=0, max_value=60, step=1)
    gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0, step=0.1)
    berat = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=40.0, step=0.1)
    submit = st.form_submit_button("Prediksi")

# Ketika tombol ditekan
if submit:
    try:
        # Buat dataframe input
        input_data = pd.DataFrame([{
            'Usia': usia,
            'Gender': gender_mapping[gender],
            'TinggiBadan': tinggi,
            'BeratBadan': berat
        }])

        # Prediksi
        pred = model.predict(input_data)[0]
        pred_label = status_mapping[pred]
        probas = model.predict_proba(input_data)[0]

        # Hasil
        st.success(f"✅ Hasil Prediksi: **{pred_label}**")
        st.write("Probabilitas masing-masing kelas:")
        for i, cls in status_mapping.items():
            st.write(f"- {cls}: {probas[i]*100:.2f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
