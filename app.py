# app.py
import streamlit as st
from backend import generate_answer

# -------------------------
# UI CONFIG
# -------------------------
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Chatbot")
st.caption("AI-powered medical assistant (Educational use only)")

# -------------------------
# CHAT HISTORY
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
user_query = st.chat_input("Ask a medical question...")

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(user_query)
            st.markdown("### 🧠 Answer")
    st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


st.markdown("---")
st.warning("⚠️ This chatbot is for educational purposes only. Always consult a doctor.")