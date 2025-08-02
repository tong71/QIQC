import streamlit as st
import google.generativeai as genai

st.title("AI自動生成Likert問卷題目 (Gemini)")

# API Key 設定
genai.configure(api_key=st.secrets['gemini']['api_key'])

subject = st.text_input("請輸入問卷主題（如：學習動機、工作壓力）")
factors = st.text_input("請輸入要測量的因子名稱（多個用逗號分隔，如：自信心, 壓力調適, 動機）")
questions_per_factor = st.number_input("每個因子需要幾題？", min_value=1, max_value=10, value=3, step=1)
btn = st.button("自動生成Likert題目")

def generate_with_gemini(subject, factor, n):
    prompt = (
        f"請用繁體中文，以「{subject}」為主題，為「{factor}」這個因子生成{n}個Likert問卷題目，每題用第一人稱陳述句（例如「我覺得…」），適合心理測驗或社會科學問卷，每題單獨一行，勿加說明。"
    )
    model = genai.GenerativeModel('gemini-1.5-pro')  # 或 gemini-pro
    response = model.generate_content(prompt)
    # 依照 API 回傳格式擷取內容
    if hasattr(response, "text"):
        return [line for line in response.text.split('\n') if line.strip()]
    else:
        return ["API 回傳格式有誤，請檢查"]

if btn and subject and factors:
    st.info("AI生成中，請稍候...")
    all_questions = []
    for factor in [f.strip() for f in factors.split(',') if f.strip()]:
        items = generate_with_gemini(subject, factor, questions_per_factor)
        st.markdown(f"**{factor}** 因子題目：")
        for q in items:
            st.write(q)
        all_questions.extend(items)
    st.success("題目生成完畢！可複製到出題模式進行語意分析、重複題檢查。")
