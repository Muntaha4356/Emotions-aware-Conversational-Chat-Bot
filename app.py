import os
import json
import numpy as np
import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from huggingface_hub import hf_hub_download
import google.generativeai as genai

MODEL_REPO = "muntaha123/emotion-clever-model"
DATA_REPO = "muntaha123/emotion-clever-dataset"
CANDIDATE_CONDITIONS = [
    "grief", "anxiety", "chronic illness", "isolation",
    "financial stress", "relationship stress", "general wellbeing",
]

st.set_page_config(
    page_title="Emotion-Aware Support Chatbot",
    page_icon="\U0001F4AC",
    layout="centered",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        max-width: 740px;
    }

    .app-header {
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .app-header p {
        color: #8a8f98;
        font-size: 0.92rem;
        margin-top: 0;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.6rem 0.9rem;
    }

    .pill-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 4px;
        margin-bottom: 2px;
    }
    .pill {
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 999px;
        background: #f0f1f5;
        color: #55595e;
        border: 1px solid #e3e5ea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>Emotion-Aware Support Chatbot</h1>
        <p>Detects how you're feeling, finds grounded information, responds with care.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Setting things up — this only happens once...")
def load_everything():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model.eval()

    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    chunks_path = hf_hub_download(DATA_REPO, "chunks.json", repo_type="dataset")
    thresholds_path = hf_hub_download(DATA_REPO, "thresholds.json", repo_type="dataset")
    labels_path = hf_hub_download(DATA_REPO, "label_names.json", repo_type="dataset")

    with open(chunks_path) as f:
        all_chunks = json.load(f)
    with open(thresholds_path) as f:
        thresholds = json.load(f)
    with open(labels_path) as f:
        label_names = json.load(f)

    embedder = SentenceTransformer("all-mpnet-base-v2")
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    tokenized_corpus = [t.lower().split() for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return model, tokenizer, zero_shot, all_chunks, thresholds, label_names, embedder, faiss_index, bm25


model, tokenizer, zero_shot, all_chunks, thresholds, label_names, embedder, faiss_index, bm25 = load_everything()

gemini_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
genai.configure(api_key=gemini_key)
llm = genai.GenerativeModel("gemini-2.0-flash")


def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).numpy()[0]
    detected = [label_names[i] for i in range(len(probs)) if probs[i] > thresholds[i]]
    return detected if detected else ["neutral"]


def infer_condition(text):
    result = zero_shot(text, CANDIDATE_CONDITIONS)
    return result["labels"][0], result["scores"][0]


def optimize_query(text):
    prompt = (
        "Rewrite this message into a short, information-dense search query. "
        "Remove filler words and hedging. Output ONLY the rewritten query.\n"
        f'Message: "{text}"\nQuery:'
    )
    return llm.generate_content(prompt).text.strip()


def bm25_search(query, top_k=20):
    scores = bm25.get_scores(query.lower().split())
    return list(np.argsort(scores)[::-1][:top_k])


def dense_search(query, top_k=20, condition_filter=None):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    search_k = top_k * 3 if condition_filter else top_k
    _, indices = faiss_index.search(q_emb, search_k)
    indices = indices[0].tolist()
    if condition_filter:
        indices = [i for i in indices if all_chunks[i]["condition"] == condition_filter][:top_k]
    return indices


def hybrid_retrieve(query, condition=None, top_k=3, k_rrf=60):
    bm25_indices = bm25_search(query, top_k=20)
    dense_indices = dense_search(query, top_k=20, condition_filter=condition)

    rrf_scores = {}
    for rank, idx in enumerate(bm25_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
    for rank, idx in enumerate(dense_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)

    top_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [all_chunks[idx] for idx, _ in top_indices]


def generate_response(user_text, emotion, condition, retrieved_chunks):
    context = "\n\n".join(c["text"] for c in retrieved_chunks)
    prompt = f"""You are a supportive assistant. User's message: "{user_text}"
Detected emotion: {emotion}
Detected topic: {condition}
Grounded information (use only this, don't invent facts or numbers):
{context}

Write a warm, empathetic response (3-5 sentences) that acknowledges the {emotion} emotion first,
then draws on the grounded information above. Avoid sounding clinical."""
    return llm.generate_content(prompt).text.strip()


def chatbot_respond(user_text):
    emotions = predict_emotions(user_text)
    condition, condition_conf = infer_condition(user_text)
    query = optimize_query(user_text)
    retrieved = hybrid_retrieve(query, condition=condition, top_k=3)
    response = generate_response(user_text, emotions[0], condition, retrieved)
    return {
        "emotions": emotions,
        "condition": condition,
        "condition_confidence": condition_conf,
        "response": response,
    }


if "history" not in st.session_state:
    st.session_state.history = []

# ---- Render conversation ----
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["response"])
        st.markdown(
            f"""<div class="pill-row">
                <span class="pill">{', '.join(turn['emotions'])}</span>
                <span class="pill">{turn['condition']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

# ---- Input ----
user_input = st.chat_input("What's on your mind?")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot_respond(user_input)
        st.write(result["response"])
        st.markdown(
            f"""<div class="pill-row">
                <span class="pill">{', '.join(result['emotions'])}</span>
                <span class="pill">{result['condition']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.session_state.history.append({
        "user": user_input,
        "response": result["response"],
        "emotions": result["emotions"],
        "condition": result["condition"],
    })
