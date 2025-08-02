import streamlit as st

def app():
    st.title("KMO 指數 (Kaiser-Meyer-Olkin Test)")
    st.markdown("""
    - **KMO 指數**用來評估資料是否適合做因子分析。
    - **KMO > 0.6** 通常認為適合進行因子分析。
    - 適用於檢查變項間相關性是否足夠。
    """)
