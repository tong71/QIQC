import streamlit as st

def app():
    st.title("t-SNE & UMAP（降維視覺化）")
    st.markdown("""
    - **t-SNE** 與 **UMAP** 都是高維資料的降維視覺化方法。
    - 可用於顯示填答者分佈、聚集與離群者。
    - 在異常偵測結果後，視覺化分群／異常點。
    """)
