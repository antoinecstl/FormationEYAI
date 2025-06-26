from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

import ollama  # pip install ollama-python

# --------------------- UI & style -----------------------------------------
st.set_page_config(page_title="🤖 RAG PDF Chat", page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
html, body {
    font-family: 'Helvetica Neue', sans-serif;
    background-color: #f4f7fa;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
}
.chat-container {
    max-width: 1200px;
    margin: auto;
    padding: 1rem;
}
.user-msg {
    background: #007bff;
    color: white;
    border-radius: 20px 20px 0px 20px;
    padding: 1rem;
    margin: 0.5rem 0;
    align-self: flex-end;
    max-width: 100%;
    word-wrap: break-word;
}
.bot-msg {
    background: #e9ecef;
    color: #212529;
    border-radius: 20px 20px 20px 0px;
    padding: 1rem;
    margin: 0.5rem 0;
    align-self: flex-start;
    max-width: 100%;
    word-wrap: break-word;
}
.chat-area {
    display: flex;
    flex-direction: column;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------- Config ---------------------------------------------
MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
DOC_TOP_K = 3
CHUNK_TOP_K = 5
CANDIDATES_K = 20
NEIGHBORS = 1
LAMBDA_DIVERSITY = 0.3
SIM_THRESHOLD = 0.25
TEMPERATURE = 0.2
MAX_TOKENS = 2048

# --------------------- Helpers LLM ----------------------------------------

def _call_llm(messages: List[Dict[str, str]], *, temperature: float = 0.1, max_tokens: int = 2048, stream: bool = False):
    return ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=stream,
        options={"temperature": temperature, "num_predict": max_tokens},
    )


def embed_texts(texts: List[str]) -> np.ndarray:
    return np.array([ollama.embeddings(model=EMBEDDING_MODEL, prompt=t)["embedding"] for t in texts], dtype="float32")

# --------------------- PDF utils -----------------------------------------

def clean_text(text: str) -> str:
    import re
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_pdf_text(path: str) -> str:
    return clean_text("\n".join(page.extract_text() or "" for page in PdfReader(path).pages))

# --------------------- Chunking & résumé ----------------------------------

def auto_chunk_size(tokens: int) -> int:
    return 1024 if tokens < 8000 else 768 if tokens < 20000 else 512


def chunk_document(text: str) -> List[str]:
    size = auto_chunk_size(len(text.split()))
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". "],
        chunk_size=size,
        chunk_overlap=int(size*0.25),
        length_function=len,
    )
    return [c for c in splitter.split_text(text) if len(c) > 100]


def make_summary(text: str) -> str:
    messages = [
        {"role": "system", "content": "Vous êtes un expert en synthèse documentaire. Résumez le texte suivant en trois parties : (1) Contexte, (2) Points clés, (3) Conclusions. Répondez en français."},
        {"role": "user", "content": text[:120000]}
    ]
    return _call_llm(messages)["message"]["content"].strip()

# --------------------- Index hiérarchique ---------------------------------
class RagIndex:
    def __init__(self):
        self.doc_index: Optional[faiss.IndexFlatIP] = None
        self.chunk_index: Optional[faiss.Index] = None
        self.doc_meta: List[Dict[str, Any]] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.chunk_emb: Optional[np.ndarray] = None

    def build(self, uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
        doc_embs, chunk_embs_list = [], []
        for doc_id, uf in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getbuffer())
                path = tmp.name
            full_text = extract_pdf_text(path)
            os.unlink(path)

            summary = make_summary(full_text)
            self.doc_meta.append({"filename": uf.name, "summary": summary})
            doc_embs.append(embed_texts([summary])[0])

            chunks = chunk_document(full_text)
            chunk_embs = embed_texts(chunks)
            chunk_embs_list.append(chunk_embs)
            for i, txt in enumerate(chunks):
                self.chunk_meta.append({"doc_id": doc_id, "text": txt, "chunk_id": i})

        self.doc_index = faiss.IndexFlatIP(len(doc_embs[0]))
        self.doc_index.add(np.vstack(doc_embs).astype("float32"))

        self.chunk_emb = np.vstack(chunk_embs_list).astype("float32")
        self.chunk_index = faiss.IndexHNSWFlat(self.chunk_emb.shape[1], 32)
        self.chunk_index.add(self.chunk_emb)

    def _is_global(self, query: str, thr: float = 0.78) -> bool:
        examples = [
            "De quoi parle ce document ?",
            "Quel est le sujet principal ?",
            "Fais un résumé du document",
        ]
        emb_q = embed_texts([query])[0]
        emb_ex = embed_texts(examples)
        sims = emb_ex @ emb_q / (np.linalg.norm(emb_ex, axis=1) * np.linalg.norm(emb_q) + 1e-6)
        return float(np.max(sims)) >= thr

    def _mmr(self, q: np.ndarray, cand: np.ndarray, k: int) -> List[int]:
        selected, rest = [], list(range(len(cand)))
        while len(selected) < min(k, len(rest)):
            best, best_score = None, -1e9
            for idx in rest:
                sim_q = float(q @ cand[idx] / (np.linalg.norm(q) * np.linalg.norm(cand[idx]) + 1e-6))
                sim_s = max(cosine_similarity(cand[idx][None, :], cand[selected])[0]) if selected else 0.
                score = LAMBDA_DIVERSITY*sim_q - (1-LAMBDA_DIVERSITY)*sim_s
                if score > best_score:
                    best, best_score = idx, score
            selected.append(best)
            rest.remove(best)
        return selected

    def retrieve(self, query: str) -> Tuple[List[str], List[int], Optional[str]]:
        q_emb = embed_texts([query])[0]
        _, I_doc = self.doc_index.search(q_emb[None, :], DOC_TOP_K)
        allowed = set(I_doc[0])

        mask = [i for i, m in enumerate(self.chunk_meta) if m["doc_id"] in allowed]
        sub_emb = self.chunk_emb[mask]
        sub_idx = faiss.IndexFlatIP(sub_emb.shape[1])
        sub_idx.add(sub_emb)
        D, I = sub_idx.search(q_emb[None, :], min(CANDIDATES_K, len(mask)))
        pool = [mask[idx] for idx in I[0]]
        pool = [idx for idx, d in zip(pool, D[0]) if 1-d <= SIM_THRESHOLD] or [mask[I[0][0]]]

        cand_emb = self.chunk_emb[pool]
        selected = [pool[i] for i in self._mmr(q_emb, cand_emb, CHUNK_TOP_K)]
        expanded = {j for idx in selected for j in range(idx-NEIGHBORS, idx+NEIGHBORS+1)}
        final = [i for i in expanded if 0 <= i < len(self.chunk_meta)][:CHUNK_TOP_K]

        contexts = [self.chunk_meta[i]["text"] for i in final]
        summary = self.doc_meta[int(I_doc[0][0])]["summary"] if self._is_global(query) else None
        return contexts, final, summary

# ---------------- Prompt builder -----------------------------------------
def build_prompt(question: str, contexts: List[str], summary: Optional[str], history: List[Dict[str, str]] = None):
    ctx_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    if summary:
        ctx_block = f"[Résumé] {summary}\n\n" + ctx_block
    
    system = (
        "Vous êtes un assistant expert. Utilisez uniquement les informations suivantes pour répondre en français. "
        "Citez vos sources avec les balises [n]. Si l'information n'est pas trouvée, informez-en l'utilisateur."
    )
    
    messages = [{"role": "system", "content": system}]
    
    if history and len(history) > 0:
        previous_messages = history[:-1] if history[-1]["role"] == "user" else history
        messages.extend(previous_messages)
    
    current_user_content = f"CONTEXTE(S):\n{ctx_block}\n\nQUESTION: {question}\n\nRéponse:"
    messages.append({"role": "user", "content": current_user_content})
    
    return messages

# ---------------- Etat Streamlit ----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# ---------------- Sidebar upload ----------------------------------------
with st.sidebar:
    st.header("📚 Documents")
    files = st.file_uploader("Déposez vos PDF", type=["pdf"], accept_multiple_files=True)
    if st.button("🔄 Réinitialiser"):
        st.session_state.clear()
        st.rerun()

# ---------------- Index construction ------------------------------------
if files and st.session_state.rag is None:
    with st.spinner("📄 Indexation en cours…"):
        rag = RagIndex()
        rag.build(files)
        st.session_state.rag = rag
    st.success(f"{len(files)} document(s) indexé(s) ! Posez vos questions.")

# ---------------- Chat display -----------------------------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
if not st.session_state.messages:
    if st.session_state.rag is not None:
        st.markdown(
            """
            <div class="bot-msg">
            👋 Bonjour ! Je suis votre assistant IA documentaire.
            <br>Posez-moi une question sur le contenu de vos documents.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="bot-msg">
            👋 Bienvenue dans RAG PDF Chat !
            <br>Commencez par télécharger un ou plusieurs documents PDF dans le panneau latéral.
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.markdown("<div class='chat-area'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        css = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Chat input -------------------------------------------
query = st.chat_input("Votre question…", disabled=st.session_state.processing or st.session_state.rag is None)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    # Réafficher l'entrée de l'utilisateur dans le style amélioré
    st.markdown(f'<div class="user-msg">{query}</div>', unsafe_allow_html=True)
    st.session_state.processing = True  # <- correctif

    rag: RagIndex = st.session_state.rag  # type: ignore
    contexts, indices, summary = rag.retrieve(query)
    
    prompt = build_prompt(query, contexts, summary, st.session_state.messages)

    placeholder = st.empty()
    collected_parts: List[str] = []

    for chunk in _call_llm(prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=True):
        token = chunk["message"]["content"]
        collected_parts.append(token)
        placeholder.markdown(f'<div class="bot-msg">{"".join(collected_parts)}</div>', unsafe_allow_html=True)

    full_answer = "".join(collected_parts)
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state.processing = False

    with st.expander("🔍 Contextes"):
        for i, ctx in enumerate(contexts):
            st.text_area(f"[{i+1}]", ctx, height=120)
