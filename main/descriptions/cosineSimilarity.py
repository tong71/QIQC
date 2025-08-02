import streamlit as st

def app():
    st.title("Cosine Similarity（餘弦相似度）")
    st.markdown("""
    - 用於量化兩個向量的相似程度，範圍 -1 ~ 1。
    - 問卷語意分析中，計算題目間語意重疊。
    - **值越高代表語意越接近。**
    """)
    st.latex(r'''Cosine\ Similarity = \frac{\vec{A}\cdot\vec{B}}{\|\vec{A}\|\|\vec{B}\|}''')
