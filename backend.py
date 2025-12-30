# backend.py
import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from transformers import pipeline

# -------------------------
# CONFIG
# -------------------------
INDEX_NAME = "medicalbot"

# -------------------------
# EMBEDDINGS
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# VECTORSTORE (LOAD EXISTING INDEX)
# -------------------------
vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# LLM
# -------------------------
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)

# -------------------------
# CLEANING HELPERS
# -------------------------
def clean_text(text: str) -> str:
    """
    Clean retrieved PDF text
    """
    text = re.sub(r"/C\d+", "", text)     # remove /C15, /C12 etc
    text = re.sub(r"\s+", " ", text)      # remove extra spaces
    return text.strip()


def clean_answer(text: str) -> str:
    """
    Clean model output
    """
    text = re.sub(r"/C\d+", "", text)
    text = text.replace("Answer:", "").strip()
    return text


# -------------------------
# CONTEXT RETRIEVAL
# -------------------------
def get_context_from_pinecone(query: str) -> str:
    docs = retriever.invoke(query)

    cleaned_chunks = []
    for doc in docs:
        cleaned_chunks.append(clean_text(doc.page_content))

    return "\n\n".join(cleaned_chunks)


# -------------------------
# ANSWER GENERATION
# -------------------------
def generate_answer(query: str) -> str:
    context = get_context_from_pinecone(query)

    prompt = f"""
You are a medical assistant.

Answer the question clearly, accurately, and in a patient-friendly manner.
Use simple medical language.

Rules:
- Do NOT copy sentences directly from the context
- Do NOT repeat information
- Do NOT include document codes or references
- Structure the answer properly with headings or bullet points
- Keep the answer concise but complete

If relevant, include:
• Definition
• Causes
• Symptoms
• Treatment
• When to see a doctor

Context:
{context}

Question:
{query}

Answer:
"""

    raw_output = qa_pipeline(prompt)[0]["generated_text"]

    return clean_answer(raw_output)




# activate env
# env\Scripts\activate

# run streamlit
# python -m streamlit run app.py