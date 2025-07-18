import streamlit as st
import pandas as pd
import graphviz
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


st.title("Visualisasi Pohon Keputusan - Deteksi Stunting")


uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Awal")
    st.dataframe(df.head())

    le_gender = LabelEncoder()
    le_status = LabelEncoder()
    df['Jenis Kelamin'] = le_gender.fit_transform(df['Jenis Kelamin'])
    df['Status Gizi'] = le_status.fit_transform(df['Status Gizi'])

    X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
    y = df['Status Gizi']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)

    # Buat visualisasi pohon keputusan
    class_names = [str(c) for c in le_status.classes_]
    feature_names = ['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']

    dot_data = export_graphviz(
        dt_model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True, special_characters=True
    )

    # Tampilkan visualisasi
    st.subheader("Visualisasi Pohon Keputusan")
    st.graphviz_chart(dot_data)

else:
    st.info("Silakan upload file CSV berisi data stunting.")
