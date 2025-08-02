import streamlit as st

def app():
    st.title("Isolation Forest（孤立森林）")
    st.markdown("""
    - 一種**異常偵測機器學習演算法**。
    - 適用於樣本數中等（30-100份）的問卷異常填答檢測。
    - 通過隨機切分特徵空間，快速識別離群樣本。
    """)
