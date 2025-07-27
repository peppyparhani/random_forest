import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


data = {
    'Jenis_Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
    'Usia_Bulan': [24, 36, 18, 48, 30, 12],
    'Tinggi_Badan': [80, 85, 72, 90, 78, 65],
    'Berat_Badan': [10, 12, 8, 14, 9, 6],
    'Status': ['Stunting', 'Normal', 'Sangat Stunting', 'Normal', 'Stunting', 'Sangat Stunting']
}
df = pd.DataFrame(data)

le_gender = LabelEncoder()
df['Jenis_Kelamin'] = le_gender.fit_transform(df['Jenis_Kelamin'])

le_status = LabelEncoder()
df['Status'] = le_status.fit_transform(df['Status'])

X = df[['Jenis_Kelamin', 'Usia_Bulan', 'Tinggi_Badan', 'Berat_Badan']]
y = df['Status']
model = DecisionTreeClassifier()
model.fit(X, y)

st.set_page_config(page_title="Prediksi Stunting Balita", layout="centered")
st.title("📊 Prediksi Status Stunting Balita")

st.markdown("Masukkan data balita untuk mengetahui status stunting berdasarkan model Decision Tree.")


gender = st.selectbox("Jenis Kelamin", le_gender.classes_)
usia = st.slider("Usia (bulan)", 0, 60, 24)
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=80.0)
berat = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, value=10.0)

if st.button("Prediksi Status"):
    gender_encoded = le_gender.transform([gender])[0]
    data_input = [[gender_encoded, usia, tinggi, berat]]
    prediksi = model.predict(data_input)[0]
    label = le_status.inverse_transform([prediksi])[0]

    st.success(f"🧒 Status balita diprediksi sebagai: **{label}**")

    
    if label == 'Normal':
        st.balloons()
    elif label == 'Stunting':
        st.info("Balita mengalami stunting ringan. Cek pola makan dan konsultasikan ke posyandu.")
    else:
        st.warning("⚠️ Balita mengalami stunting berat. Segera konsultasi ke ahli gizi atau dokter.")

