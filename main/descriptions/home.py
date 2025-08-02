import streamlit as st

def app():
    # 首頁主內容
    st.title("【Likert-scale】 問卷智能質檢系統 QIQC")
    st.markdown("""
    ### 系統簡介
    「題目先防錯、填答後揪錯」——QIQC 結合 NLP、統計檢定與異常偵測，協助研究者設計高品質 Likert 型問卷，並對填答資料進行智能質量檢查。
    <br><br>
    **QIQC 核心流程：**
    - 出題模式：語意分析、重複題預防（避免題目過於相似）
    - 驗證模式：資料檢驗、異常填答偵測（保證問卷可靠性）
    <br>
    <br>
    點選左側各模式或名詞可了解更多！
    """, unsafe_allow_html=True)
    st.info("開源專案網址：[QIQC on GitHub](https://github.com/你的專案網址)")
