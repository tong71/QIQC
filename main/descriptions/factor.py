import streamlit as st

def app():
    st.title("因子分析（Factor Analysis）")
    st.markdown("""
    - 探討多題目間潛在的「共通結構」。
    - 將大量變項歸納成較少的「因子」。
    - 常見應用：人格量表、心理測驗等。
    """)
    st.info("範例：MBTI 分為 4 個因子；五大人格特質分為 5 個因子")
