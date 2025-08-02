import streamlit as st
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import umap
st.title("驗證模式：資料檢驗與異常填答偵測")
st.markdown("""
依照下列步驟，進行問卷資料驗證及異常填答檢測，提升量表科學性與資料品質。
""")

# ===== Step 1: 上傳與設定 =====
st.header("Step 1. 上傳問卷資料與設定")
file = st.file_uploader("請上傳問卷答案檔案 (CSV 或 Excel)", type=["csv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("你上傳的表格：")
    st.dataframe(df, hide_index=True)

    # 只分析純數值型欄位
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("未發現任何純數值型欄位，請確認你的問卷答案檔案格式正確（應該每題都是數值分數）！")
        st.stop()

    question_cols = st.multiselect(
        "請選擇要分析的問卷題目欄位（僅可選純數值型欄位）",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
    if not question_cols or len(question_cols) < 2:
        st.warning("請選取至少兩個純數值型欄位。")
        st.markdown("#### 範例資料格式")
        example = pd.DataFrame({
            "Q1": [1, 2, 4, 5],
            "Q2": [3, 1, 2, 4],
            "Q3": [5, 4, 3, 2],
        })
        st.dataframe(example, hide_index=True)
        st.markdown("""
        <small>
        - 每一欄代表一個題目，欄位名稱如 Q1、Q2...<br>
        - 每一列是一份問卷答案<br>
        - 所有欄位都應為數字（如 1~5、1~7 之 Likert 分數）<br>
        - 若欄位中混有文字（如 "非常同意"），請先轉為數值
        </small>
        """, unsafe_allow_html=True)
        st.stop()


    # 資料型態轉換與缺失值處理
    dfq = df[question_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n_questions = len(question_cols)
    n_samples = dfq.shape[0]
    st.write(f"✔️ 有效樣本數：{n_samples}，題目數：{n_questions}")

    if n_samples < 5:
        st.warning("樣本數過少，無法進行後續分析。")
        st.stop()

    user_n_factors = st.number_input("預期因子數", min_value=1, max_value=n_questions, value=2)

    # ===== Step 2: KMO/Bartlett =====
    st.header("Step 2. 因子分析適配性檢定 (KMO / Bartlett’s Test)")
    try:
        kmo_all, kmo_model = calculate_kmo(dfq)
        bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(dfq)
        st.write(f"KMO 指數: **{kmo_model:.2f}**")
        st.write(f"Bartlett’s 檢定 p-value: **{bartlett_p:.4f}**")
        if kmo_model < 0.6:
            st.warning("KMO 指數 < 0.6，不建議進行因子分析！")
        if bartlett_p >= 0.05:
            st.warning("Bartlett’s p-value >= 0.05，資料無法證明具因子結構！")
        if (kmo_model >= 0.6) and (bartlett_p < 0.05):
            st.success("✔️ 通過 KMO 與 Bartlett’s Test，適合因子分析。")
    except Exception as e:
        st.error(f"KMO/Bartlett 檢定無法計算，請確認資料格式（錯誤訊息：{e}）")
        st.stop()

    # ===== Step 2.5: Scree Plot, Eigenvalue, 累積變異量 =====
    st.header("Step 2.5. 建議最適因子數")
    try:
        fa = FactorAnalyzer()
        fa.fit(dfq)
        ev, v = fa.get_eigenvalues()
        fig, ax = plt.subplots()
        sns.lineplot(x=np.arange(1, len(ev)+1), y=ev, marker='o', ax=ax)
        ax.axhline(1, color='r', ls='--', label='Eigenvalue=1')
        ax.set_title('Scree Plot')
        ax.set_xlabel('因子數')
        ax.set_ylabel('Eigenvalue')
        ax.legend()
        st.pyplot(fig)

        st.write("各因子的 Eigenvalue：", [f"{x:.2f}" for x in ev])
        cumulative_var = np.cumsum(fa.get_factor_variance()[1])
        st.write("累積解釋變異量：", [f"{x:.2%}" for x in cumulative_var])

        elbow = np.argmax(ev < 1) + 1
        suggested_n_factors = max(elbow, np.argmax(cumulative_var >= 0.6)+1)
        st.info(f"建議因子數：{suggested_n_factors}（Elbow/Eigenvalue>1/累積變異量>60% 綜合判斷）")

        use_suggested = st.radio("要採用建議因子數嗎？", ["採用建議", "保持原設定"])
        final_n_factors = suggested_n_factors if use_suggested == "採用建議" else user_n_factors
    except Exception as e:
        st.error(f"因子數建議計算失敗：{e}")
        st.stop()

    # ===== Step 3: 執行因子分析 =====
    st.header("Step 3. 執行因子分析")
    try:
        fa = FactorAnalyzer(n_factors=int(final_n_factors), rotation="varimax")
        fa.fit(dfq)
        loadings = pd.DataFrame(fa.loadings_, index=question_cols, columns=[f"因子{i+1}" for i in range(int(final_n_factors))])
        st.write("題目因子載荷 (Loadings)：")
        st.dataframe(loadings.style.background_gradient(cmap="YlGnBu"), hide_index=True)
    except Exception as e:
        st.error(f"因子分析執行失敗，請檢查資料（錯誤訊息：{e}）")
        st.stop()

    # 標記異常題項
    issues = []
    for idx, row in loadings.iterrows():
        primary = row.abs().max()
        cross = sum(row.abs() > 0.4)
        if primary < 0.3:
            issues.append((idx, "載荷值低 (<0.3)"))
        if cross > 1:
            issues.append((idx, "跨因子載荷 > 0.4"))
    if issues:
        st.warning("⚠️ 以下題目載荷異常，建議人工檢查：")
        for idx, reason in issues:
            st.write(f"- {idx}：{reason}")
    else:
        st.success("所有題目載荷表現正常！")

    # ===== Step 4: Cronbach’s α =====
    st.header("Step 4. 一致性檢驗（Cronbach’s α）")
    for i in range(int(final_n_factors)):
        items = loadings.index[loadings.iloc[:, i].abs() >= 0.3].tolist()
        if len(items) >= 2:
            subscale = dfq[items]
            corr = subscale.corr().values
            n = subscale.shape[1]
            avg_inter_corr = (np.sum(corr)-n)/(n*(n-1))
            alpha_val = n*avg_inter_corr / (1+(n-1)*avg_inter_corr)
            st.write(f"因子{i+1} Cronbach’s α：**{alpha_val:.2f}** （{', '.join(items)}）")
        else:
            st.write(f"因子{i+1} Cronbach’s α 無法計算（有效題目數不足）")

    # ===== Step 5 & 6: 異常填答自動模型選擇 =====
    st.header("Step 5. 異常填答檢測模型自動選擇")
    model_used = None
    anomaly_score = None
    threshold = None
    if (n_questions > 10) and (n_samples > 100):
        model_used = "Autoencoder (需另行客製實作)"
        st.warning("Autoencoder 屬深度學習模型，需客製，這裡預設用 Isolation Forest 示範")
        # 預設用 Isolation Forest 以方便測試
    elif (5 <= n_questions <= 20) and (30 <= n_samples <= 100):
        model_used = "Isolation Forest"
    elif n_samples < 30:
        model_used = None
        st.info("樣本過少，不進行異常填答檢測")
    else:
        model_used = "Isolation Forest"
    if model_used == "Isolation Forest":
        X = StandardScaler().fit_transform(dfq)
        iso = IsolationForest(random_state=42)
        preds = iso.fit_predict(X)
        anomaly_score = -iso.decision_function(X)
        st.success("已自動選用 Isolation Forest 異常檢測模型")
    elif model_used == "Autoencoder (需另行客製實作)":
        st.info("Autoencoder 模型未預設實作，需自訂。")
    if anomaly_score is not None:
        st.write("各樣本異常分數（越高越異常）：")
        st.dataframe(pd.DataFrame({"異常分數": anomaly_score}), hide_index=True)
        threshold = np.percentile(anomaly_score, 95)
        st.write(f"建議異常門檻（95百分位）：**{threshold:.2f}**")
        anomaly_flag = anomaly_score > threshold

        # 直方圖
        fig2, ax2 = plt.subplots()
        sns.histplot(anomaly_score, bins=20, ax=ax2, color="blue")
        ax2.axvline(threshold, color="red", linestyle="--", label="異常門檻")
        ax2.set_title("異常分數分布")
        ax2.legend()
        st.pyplot(fig2)

        # ===== Step 7: 降維視覺化 (UMAP) =====
        st.header("Step 6. 降維視覺化與異常樣本標記 (UMAP)")
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(X)
        plt.figure(figsize=(6,4))
        plt.scatter(embedding[:,0], embedding[:,1], c=anomaly_flag, cmap="coolwarm", s=40, alpha=0.8)
        plt.title("填答樣本降維視覺化（紅=疑似異常）")
        st.pyplot(plt)
        st.write("點選表格可進一步查看疑似異常填答者原始資料。")
else:
    st.info("請先上傳問卷資料檔案。")
