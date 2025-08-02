import streamlit as st

def app():
    st.title("Cronbach’s α（克朗巴哈信度係數）")
    st.markdown("""
    - 衡量「同一因子」內題目的內部一致性。
    - **α值越高（一致性越好）**
    - 通常α > 0.7 被視為問卷可靠。
    """)
    st.latex(r'''\alpha = \frac{N \cdot \bar{c}}{\bar{v} + (N-1)\bar{c}}''')
