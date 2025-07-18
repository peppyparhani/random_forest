import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    # Baca dataset
    df = pd.read_csv("data_balita.csv")

    # Encode kolom kategori
    le_gender = LabelEncoder()
    le_status = LabelEncoder()

    df['Jenis Kelamin'] = le_gender.fit_transform(df['Jenis Kelamin'])
    df['Status Gizi'] = le_status.fit_transform(df['Status Gizi'])

    # Fitur dan target
    X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
    y = df['Status Gizi']

    # Model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_status, le_gender, X.columns.tolist()

# Load model dan encoder
model, le_status, le_gender, feature_names = load_model()

# =============================
# UI Streamlit
# =============================
st.set_page_config(page_title="Deteksi Stunting", layout="centered")
st.title("🧒 Deteksi Stunting Balita dengan Random Forest")

st.markdown("Masukkan data balita untuk memprediksi status gizinya:")

# Input dari pengguna
umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=80.0)

if st.button("🔍 Prediksi Status Gizi"):
    try:
        # Encode input
        jk_encoded = le_gender.transform([jenis_kelamin])[0]
        input_data = [[umur, jk_encoded, tinggi]]

        # Prediksi kelas & probabilitas
        hasil_encoded = model.predict(input_data)[0]
        hasil = le_status.inverse_transform([hasil_encoded])[0]
        probs = model.predict_proba(input_data)[0]

        # Tampilkan hasil prediksi
        st.success(f"🌟 Status Gizi Balita: **{hasil.upper()}**")

        # ================================
        # 🔢 Probabilitas sebagai bar chart
        # ================================
        st.subheader("📊 Probabilitas Prediksi")
        prob_df = pd.DataFrame({
            'Status Gizi': le_status.inverse_transform(model.classes_),
            'Probabilitas': probs
        })
        st.bar_chart(prob_df.set_index("Status Gizi"))

        # ================================
        # 🌳 Visualisasi salah satu pohon
        # ================================
        st.subheader("🌲 Visualisasi Salah Satu Pohon Keputusan (Tree 1 dari Random Forest)")
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_tree(model.estimators_[0], 
                  feature_names=feature_names, 
                  class_names=le_status.classes_.astype(str),
                  filled=True, rounded=True, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
