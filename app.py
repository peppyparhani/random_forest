import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Deteksi Stunting", layout="centered")
st.title("🧒 Deteksi Stunting Balita dengan Random Forest")

uploaded_file = st.file_uploader("📂 Upload File CSV Data Balita", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data dari file upload
        df = pd.read_csv(uploaded_file)

        # Encode kolom kategori
        le_gender = LabelEncoder()
        le_status = LabelEncoder()

        df['Jenis Kelamin'] = le_gender.fit_transform(df['Jenis Kelamin'])
        df['Status Gizi'] = le_status.fit_transform(df['Status Gizi'])

        # Fitur dan target
        X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
        y = df['Status Gizi']

        # Training model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        st.success("✅ Model berhasil dilatih dari file yang diunggah!")

        # ---------------------
        # Form Input Data
        # ---------------------
        st.subheader("🔢 Masukkan Data Balita untuk Prediksi")

        umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=24)
        jenis_kelamin = st.selectbox("Jenis Kelamin", le_gender.classes_)
        tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=80.0)

        if st.button("🔍 Prediksi Status Gizi"):
            jk_encoded = le_gender.transform([jenis_kelamin])[0]
            input_data = [[umur, jk_encoded, tinggi]]

            pred_encoded = model.predict(input_data)[0]
            hasil = le_status.inverse_transform([pred_encoded])[0]

            st.success(f"🌟 Status Gizi Balita: **{hasil.upper()}**")

            # Probabilitas prediksi
            st.subheader("📊 Probabilitas Prediksi")
            probs = model.predict_proba(input_data)[0]
            prob_df = pd.DataFrame({
                "Status Gizi": le_status.inverse_transform(model.classes_),
                "Probabilitas": probs
            })
            st.bar_chart(prob_df.set_index("Status Gizi"))

            # Visualisasi pohon
            st.subheader("🌲 Visualisasi Salah Satu Pohon dari Random Forest")
            fig, ax = plt.subplots(figsize=(14, 6))
            plot_tree(model.estimators_[0],
                      feature_names=X.columns,
                      class_names=le_status.classes_,
                      filled=True, rounded=True, ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
else:
    st.info("📌 Silakan unggah file CSV yang berisi data balita.")
