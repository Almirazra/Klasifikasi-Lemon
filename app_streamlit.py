import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title="belajar klasifkasi lemon",
	page_icon="🍋"
)
model=joblib.load("model_klasifikasi.lemon.joblib")
st.title("🍋 belajar klasifikasi lemon")
st.markdown("aplikasi machine learning lemon")
diameter=st.slider("Diameter",65.9,55.0,45.0)
berat=st.slider("Berat",100.0,150.0,110.0)
tebal_kulit=st.slider("Tebal Kulit",3.0,5.0,3.5)
kadar_gula=st.slider("Kadar gula",8.0,6.0,6.8)
asal_daerah=st.pills("Asal daerah",["California","Malang","Medan"],default="Medan")
warna=st.pills("warna",["Hijau pekat","Kuning kehijauan","Kuning cerah"],default="Hijau pekat")
musim_panen=st.pills("Musim Panen",["Puncak","Awal","Akhir"],default="Akhir")
if st.button("prediksi",type="primary"):
	data_baru=pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,musim_panen,warna]],columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","musim_panen","warna"])
	prediksi=model.predict(data_baru)[0]
	presentase=max(model.predict_proba(data_baru)[0])
	st.success(f"model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()
st.divider()

