# backend.py
import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from transformers import pipeline

# =========================
# CONFIG
# =========================
INDEX_NAME = "medicalbot"

# =========================
# EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# VECTORSTORE
# =========================
vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# =========================
# LLM
# =========================
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_new_tokens=450,
    temperature=0.2,
    repetition_penalty=1.3,
    do_sample=False
)

# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r"/C\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_section(text, start, end_list):
    if start not in text:
        return ""
    part = text.split(start, 1)[1]
    for end in end_list:
        if end in part:
            part = part.split(end, 1)[0]
    return part.strip()


def format_bullets(text):
    sentences = re.split(r"[.;]\s+", text)
    bullets = [f"- {s.strip()}" for s in sentences if len(s.strip()) > 20]
    return "\n".join(bullets)


# =========================
# CONTEXT
# =========================
def get_context_from_pinecone(query):
    docs = retriever.invoke(query)
    return "\n\n".join(clean_text(doc.page_content) for doc in docs)


# =========================
# ANSWER GENERATION
# =========================
def generate_answer(query: str) -> str:
    context = get_context_from_pinecone(query)

    if not context:
        return "⚠️ No relevant medical information found."

    prompt = f"""
You are a medical assistant.

Write the answer using the following sections ONLY:
Definition
Causes
Symptoms
Treatment
When to see a doctor

Keep language simple for patients.

Context:
{context}

Question:
{query}
"""

    raw = qa_pipeline(prompt)[0]["generated_text"]
    raw = clean_text(raw)

    sections = ["Definition", "Causes", "Symptoms", "Treatment", "When to see a doctor"]

    output = ""

    for i, sec in enumerate(sections):
        content = extract_section(raw, sec, sections[i+1:])
        if content:
            output += f"### {sec}\n{format_bullets(content)}\n\n"

    # 🔥 FINAL FALLBACK (never empty)
    if not output.strip():
        return raw

    return output.strip()
