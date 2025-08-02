import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("出題模式：語意分析與重複題預防")
st.markdown("""
QIQC 可協助你分析 Likert 題庫，**自動標記語意過高重疊/類似題**，提升問卷效度。  
依下列步驟操作：
""")

# Step 1: 題目輸入
st.header("Step 1. 輸入/上傳問卷題目")
input_mode = st.radio("請選擇題目來源：", ["手動輸入", "CSV/Excel 檔案上傳"])

questions = []
if input_mode == "手動輸入":
    num_questions = st.number_input("請輸入題目數量", min_value=2, max_value=50, value=5)
    questions = [st.text_input(f"題目 {i+1}", key=f"q{i}") for i in range(num_questions)]
    questions = [q for q in questions if q.strip()]
else:
    uploaded = st.file_uploader("上傳題目 CSV 或 Excel 檔", type=["csv", "xlsx"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.write("你上傳的表格：")
        st.dataframe(df, hide_index=True)
        qcol = st.selectbox("請選擇題目欄位", df.columns)
        questions = [str(q).strip() for q in df[qcol] if pd.notnull(q) and str(q).strip() != ""]


if questions and len(questions) >= 2:
    st.header("Step 2. 語意相似度分析")
    if st.button("進行語意分析"):
        with st.spinner("語意向量化及相似度計算中..."):
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embeddings = model.encode(questions)
            sim_matrix = cosine_similarity(embeddings)
            sim_df = pd.DataFrame(sim_matrix, columns=[f"Q{i+1}" for i in range(len(questions))], index=[f"Q{i+1}" for i in range(len(questions))])

        st.write("#### 題項間餘弦相似度（Cosine Similarity）")
        st.dataframe(sim_df.style.background_gradient(cmap="Blues"), use_container_width=True)

        st.header("Step 3. 自動標記潛在重複/類似題目")
        flagged = False
        for i in range(len(questions)):
            for j in range(i+1, len(questions)):
                sim = sim_matrix[i, j]
                if sim > 0.9:
                    st.warning(f"**題目 Q{i+1}：「{questions[i]}」** 與 **Q{j+1}：「{questions[j]}」** 的語意相似度為 {sim:.2f}，建議合併、刪除或重寫！")
                    flagged = True
        if not flagged:
            st.success("未發現語意過高重疊題目！")

        st.info("請對被標記的題目進行『合併、刪除或重寫』，以避免資料冗餘或心理測驗效度降低。")
else:
    st.info("請先輸入或上傳至少兩個題目。")
