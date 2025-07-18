import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Deteksi Stunting", layout="wide")
st.title("📊 Deteksi Stunting pada Balita menggunakan Decision Tree")

# Upload file CSV
uploaded_file = st.file_uploader("🗂️ Upload file CSV dataset balita", type=["csv"])

if uploaded_file is not None:
    try:
        # Membaca dataset
        df = pd.read_csv(uploaded_file)

        st.subheader("📁 Data Awal")
        st.dataframe(df.head())

        # Label Encoding
        le_gender = LabelEncoder()
        le_status = LabelEncoder()

        df['Jenis Kelamin'] = le_gender.fit_transform(df['Jenis Kelamin'])
        df['Status Gizi'] = le_status.fit_transform(df['Status Gizi'])

        # Fitur dan target
        X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
        y = df['Status Gizi']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_labels = [str(label) for label in le_status.classes_]

        st.success(f"🎯 Akurasi Model: {accuracy:.4f}")

        # Classification Report
        st.subheader("📄 Classification Report")
        report_df = pd.DataFrame(classification_report(
            y_test, y_pred, target_names=class_labels, output_dict=True
        )).transpose()
        st.dataframe(report_df)

        # Confusion Matrix
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax_cm)
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        st.pyplot(fig_cm)

        # Visualisasi Pohon Keputusan
        st.subheader("🌳 Visualisasi Decision Tree")
        fig_tree, ax_tree = plt.subplots(figsize=(16, 8))
        plot_tree(model,
                  feature_names=X.columns,
                  class_names=class_labels,
                  filled=True,
                  rounded=True,
                  fontsize=10,
                  ax=ax_tree)
        st.pyplot(fig_tree)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
else:
    st.info("💡 Silakan upload file CSV terlebih dahulu.")
