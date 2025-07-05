import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64


# Tema ayarları ve özel CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 0px 10px #ccc;
        }
        h1 {
            color: #4CAF50 !important;
        }
    </style>
""", unsafe_allow_html=True)


# 🎨 Başlık ve açıklama
st.markdown("""
    <div style="text-align:center">
        <h1 style="color:#4CAF50; font-size:42px;">🏠 Ev Kirası Tahmin Uygulaması</h1>
        <p style="font-size:18px;">Girdiğiniz bilgilere göre evin yaklaşık kirasını tahmin edin.<br>
        👩‍💻 Makine öğrenimi ile desteklenmiş kullanıcı dostu bir arayüz.</p>
    </div>
""", unsafe_allow_html=True)

# 🧠 Model ve scaler'ı yükle
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("veri_olcekleyici.pkl")

# 1. Başta geçmişi başlat
if "gecmis" not in st.session_state:
    st.session_state["gecmis"] = []

# 2. Girdileri bir form içinde al
with st.form("tahmin_formu"):
    st.subheader("📋 Bilgileri Girin")
    col1, col2 = st.columns(2)

    with col1:
        bhk = st.slider("🛏 Oda Sayısı (BHK)", 1, 10, 2)
        size = st.number_input("📐 Alan (m²)", min_value=30, max_value=500, value=100, step=10)

        bathroom = st.slider("🚿 Banyo Sayısı", 1, 5, 2)

    with col2:
        city = st.selectbox("🌆 Şehir", ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai"])
        furnishing = st.radio("🛋 Eşya Durumu", ["Furnished", "Semi-Furnished", "Unfurnished"])
        area_type = st.radio("🏗 Alan Tipi", ["Super Area", "Carpet Area", "Built Area"])

    # Formun submit butonu
    submit = st.form_submit_button("🎯 Tahmini Gör")

# 3. Eğer butona basıldıysa: tahmin yap ve geçmişe ekle
if submit:
    df = pd.DataFrame([{
        "BHK": bhk,
        "Size": size,
        "Bathroom": bathroom,
        "City": city,
        "Furnishing Status": furnishing,
        "Area Type": area_type

    }])

    df = pd.get_dummies(df)
    columns = pd.read_csv("columns.csv", header=None).iloc[:, 0].tolist()
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    scaler = joblib.load("veri_olcekleyici.pkl")
    model = joblib.load("random_forest_model.pkl")
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    tahmini_kira = np.exp(prediction[0])

    # 🎯 Geçmişe ekle
    st.session_state["gecmis"].append({
        "Şehir": city,
        "Alan": size,
        "BHK": bhk,
        "Banyo": bathroom,
        "Eşya": furnishing,
        "Kira (₹)": round(tahmini_kira, 2)
    })

    st.success(f"🎉 Tahmini Aylık Kira: ₹ {tahmini_kira:,.0f}")

# 4. Geçmişi göster
if st.session_state["gecmis"]:
    st.markdown("## 📊 Tahmin Geçmişi")
    st.dataframe(pd.DataFrame(st.session_state["gecmis"]))
