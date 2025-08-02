import streamlit as st

def app():
    st.title("出題模式：語意分析與重複題預防")
    st.markdown("""
1. **NLP語意相似分析**  
    - 使用 HuggingFace Transformers + SentenceTransformer 進行語意向量化
    - 計算問卷題項間的餘弦相似度，檢查是否有過高語意重疊
2. **主動標記潛在問題題項**  
    - 若題目相似度 > 0.9 或語意聚集度高，系統會主動標記並提醒修正
    - 使用者可根據提示合併、刪除或重寫相近題項，避免資料冗餘或效度降低
    """)
    st.info("目的：讓每個題目測量的構念更加明確，避免不同題目測量到同一概念，提升問卷效度。")
