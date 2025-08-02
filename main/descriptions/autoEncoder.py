import streamlit as st

def app():
    st.title("Autoencoder（自編碼器）")
    st.markdown("""
    - 一種**無監督深度學習模型**，常用於異常偵測。
    - 適合大樣本問卷，找出填答異常樣本。
    - 可視化特徵空間差異，協助標記異常填答者。
    """)
