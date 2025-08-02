import streamlit as st
import requests

st.title("AI自動生成Likert問卷題目 (HuggingFace API)")

# ==== 使用者輸入 ====
subject = st.text_input("請輸入問卷主題（如：學習動機、工作壓力）")
factors = st.text_input("請輸入要測量的因子名稱（多個用逗號分隔，如：自信心, 壓力調適, 動機）")
questions_per_factor = st.number_input("每個因子需要幾題？", min_value=1, max_value=10, value=3, step=1)
btn = st.button("自動生成Likert題目")

# ==== HuggingFace API 設定 ====
API_URL = "https://api-inference.huggingface.co/models/Noufy/https-api_inference.huggingface.comodels-Noufy-naive_bayes_sms"
headers = {"Authorization": f"Bearer {st.secrets['huggingface']['api_key']}"}

# 可以選擇註解掉下面這行，僅測試API狀態用
resp = requests.get(API_URL, headers=headers)
st.write("API status code: ", resp.status_code)
st.write("API response: ", resp.text)

def generate_likert_items(subject, factor, n):
    prompt = (
        f"請用繁體中文，以「{subject}」為主題，為「{factor}」這個因子生成{n}個Likert問卷題目，每題用第一人稱陳述句（例如「我覺得…」），適合心理測驗或社會科學問卷，每題單獨一行，勿加說明。"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code == 200:
        output = response.json()
        # HuggingFace 會回傳一個list，裡面每個元素是 dict
        if isinstance(output, list) and 'generated_text' in output[0]:
            text = output[0]['generated_text']
        elif isinstance(output, dict) and 'generated_text' in output:
            text = output['generated_text']
        else:
            text = str(output)
        items = [line for line in text.split('\n') if line.strip()]
        return items
    else:
        return [f"API Error: {response.status_code}", response.text]

if btn and subject and factors:
    st.info("AI生成中，請稍候...")
    all_questions = []
    for factor in [f.strip() for f in factors.split(',') if f.strip()]:
        items = generate_likert_items(subject, factor, questions_per_factor)
        st.markdown(f"**{factor}** 因子題目：")
        for q in items:
            st.write(q)
        all_questions.extend(items)
    st.success("題目生成完畢！可複製到出題模式進行語意分析、重複題檢查。")
