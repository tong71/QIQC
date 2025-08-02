import streamlit as st
import importlib
# --------------------------------------
# sidebar radio - 名詞解釋/模式說明（你自己的客製導覽）
desc_names = [
    "首頁",
    "出題模式",
    "驗證模式",
    "KMO 指數",
    "Bartlett’s 檢定",
    "Cosine Similarity",
    "因子分析（Factor Analysis）",
    "Cronbach’s α",
    "Autoencoder",
    "Isolation Forest",
    "t-SNE / UMAP"
]
desc_files = [
    "home",
    "questionMode",
    "verificationMode",
    "KMO",
    "bartlettTest",
    "cosineSimilarity",
    "factor",
    "cronbach’s-α",
    "autoEncoder",
    "isolationForest",
    "t-SNE&UMAP"
]

st.sidebar.title("網站導覽（名詞解釋與模式說明）")
side_selected = st.sidebar.radio("請選擇說明頁：", desc_names, index=0, key="side_radio")
if side_selected:
    idx = desc_names.index(side_selected)
    module_name = f"descriptions.{desc_files[idx]}"
    try:
        module = importlib.import_module(module_name)
        module.app()
    except Exception as e:
        st.sidebar.error(f"說明頁面載入失敗 ({module_name})\n\nError: {e}")
