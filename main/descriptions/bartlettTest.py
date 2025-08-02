import streamlit as st

def app():
    st.title("Bartlett’s 球形檢定 (Bartlett’s Test of Sphericity)")
    st.markdown("""
    - **Bartlett’s 檢定**檢查變項間相關矩陣是否為單位矩陣（即完全不相關）。
    - **p < 0.05** 表示資料適合進行因子分析。
    - 通常與KMO一起用來判斷資料適配性。
    """)
