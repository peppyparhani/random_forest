import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_stunting_rf.pkl')

# Judul aplikasi
st.title("Prediksi Stunting pada Balita")
st.write("Masukkan data balita untuk memprediksi status stunting (Normal, Stunting, atau Sangat Stunting).")

# Input data pengguna
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age_months = st.number_input("Usia (bulan)", min_value=0, max_value=60, value=24)
height_cm = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0, value=80.0)
weight_kg = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, value=10.0)

# Ubah input ke DataFrame sesuai nama fitur saat training
input_data = pd.DataFrame({
    "Gender": [1 if gender == "Laki-laki" else 0],
    "AgeMonths": [age_months],
    "HeightCM": [height_cm],
    "WeightKG": [weight_kg]
})

# Tombol prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data)[0]
    label_dict = {0: "Normal", 1: "Stunting", 2: "Sangat Stunting"}
    hasil = label_dict[prediction]
    st.success(f"Hasil Prediksi: **{hasil}**")
