import streamlit as st
from transformers import pipeline

st.title("AI自動生成Likert問卷題目 (本地 LLM)")

# ==== 使用者輸入 ====
subject = st.text_input("請輸入問卷主題（如：學習動機、工作壓力）")
factors = st.text_input("請輸入要測量的因子名稱（多個用逗號分隔，如：自信心, 壓力調適, 動機）")
questions_per_factor = st.number_input("每個因子需要幾題？", min_value=1, max_value=10, value=3, step=1)
btn = st.button("自動生成Likert題目")

@st.cache_resource(show_spinner="正在加載大語言模型，請稍候...")
def get_pipe():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen1.5-7B-Chat",   # 你若有更大RAM可改 "Qwen/Qwen1.5-7B-Chat"
        device_map="auto",
        trust_remote_code=True
    )

def generate_likert_items(subject, factor, n, pipe):
    prompt = (
        f"請用繁體中文，以「{subject}」為主題，為「{factor}」這個因子生成{n}個Likert問卷題目，每題以「我」開頭，"
        "請直接列出題目，每題一行，並在每題前標上阿拉伯數字序號。例如：\n1. 我覺得…\n2. 我認為…\n勿加任何說明。"
    )
    output = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    text = output[0]['generated_text']
    st.write("【DEBUG】模型原始回應：", text)  # 顯示模型原始輸出方便debug

    # 處理回應內容：移除 prompt、只留回應
    answer = text.replace(prompt, '').strip()
    items = [line for line in answer.split('\n') if line.strip()]

    st.write("【DEBUG】切分後的題目清單：", items)  # debug顯示
    return items

if btn and subject and factors:
    st.info("AI生成中，請稍候...")
    pipe = get_pipe()
    all_questions = []
    for factor in [f.strip() for f in factors.split(',') if f.strip()]:
        items = generate_likert_items(subject, factor, questions_per_factor, pipe)
        st.markdown(f"**{factor}** 因子題目：")
        for q in items:
            st.write(q)
        all_questions.extend(items)
    st.success("題目生成完畢！可複製到出題模式進行語意分析、重複題檢查。")
