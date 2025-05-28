import os
import tempfile
from typing import List, Tuple

import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

import asyncio
from ollama import AsyncClient
import ollama

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "llama3.2:3b"  # Change here if you want to try a different Ollama model
EMBEDDING_MODEL = "nomic-embed-text:latest"  # you can also set to a lighter/faster model
TOP_K = 5               # final contexts given to the LLM
CANDIDATES_K = 20       # initial retrieval set before MMR
SIM_THRESHOLD = 0.25    # cosine similarity cut‑off

st.set_page_config(page_title="RAG PDF Chat • Ollama", page_icon="🤖")
st.title("🤖📄 RAG PDF Chat – v2")

# Sidebar controls ----------------------------------------------------------------
with st.sidebar:
    st.header("Paramètres")
    temp = st.slider("Température", 0.0, 1.0, 0.1, 0.05)
    max_tokens = st.slider("Max tokens", 64, 1024, 512, 64)
    files = st.file_uploader("📚 Déposez vos PDF", type=["pdf"], accept_multiple_files=True)
    if st.button("🔄 Réinitialiser"):
        st.session_state.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def auto_chunk_size(total_tokens: int) -> int:
    """Heuristic: bigger docs need smaller chunks to fit context window."""
    # Increased minimum chunk sizes for better context
    if total_tokens < 8_000:
        return 750  # Increased from 400
    if total_tokens < 20_000:
        return 500  # Increased from 300
    return 350  # Increased from 200


def clean_text(text: str) -> str:
    """Clean extracted text to fix common PDF extraction issues."""
    import re
    # Replace multiple newlines with a single one
    text = re.sub(r'\n{2,}', '\n\n', text)
    # Replace hyphenated words at line breaks (common PDF issue)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Fix spacing issues
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def extract_pdf_text(path: str) -> str:
    """Extract and clean text from PDF."""
    raw_text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return clean_text(raw_text)


def embed_texts(texts: List[str]) -> np.ndarray:
    return np.array([
        ollama.embeddings(model=EMBEDDING_MODEL, prompt=t)["embedding"] for t in texts
    ], dtype="float32")


def build_index(docs: List[Tuple[str, str]]):
    """Split, embed & index uploaded documents. Returns FAISS index, chunks, embeddings, meta."""
    all_chunks, meta = [], []
    for fname, text in docs:
        token_est = len(text.split())  # rough proxy
        chunk_sz = auto_chunk_size(token_est)
        
        # Use better separator list that respects natural text boundaries
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", "；", "，", "。", " ", ""],
            chunk_size=chunk_sz,
            chunk_overlap=int(chunk_sz * 0.15),  # Increased overlap
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = splitter.split_text(text)
        
        # Filter out very short chunks (likely just fragments)
        chunks = [c for c in chunks if len(c) > 50]
        
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            meta.append({"file": fname, "chunk_id": idx})
    
    # Add a debug message to see chunk statistics
    if all_chunks:
        avg_len = sum(len(c) for c in all_chunks) / len(all_chunks)
        st.session_state.chunk_stats = {
            "count": len(all_chunks),
            "avg_length": avg_len,
            "min_length": min(len(c) for c in all_chunks),
            "max_length": max(len(c) for c in all_chunks),
        }
    
    emb = embed_texts(all_chunks)
    dim = emb.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(emb)
    return index, all_chunks, emb, meta


def mmr(query_emb: np.ndarray, emb: np.ndarray, top_k: int, lambda_diversity: float = 0.3):
    selected, candidates = [], list(range(len(emb)))
    while len(selected) < top_k and candidates:
        best, best_score = None, -1e9
        for idx in candidates:
            sim_to_query = float(np.dot(query_emb, emb[idx]) / (np.linalg.norm(query_emb) * np.linalg.norm(emb[idx]) + 1e-6))
            if selected:
                sim_to_selected = max(
                    cosine_similarity(emb[idx].reshape(1, -1), emb[selected])[0]
                )
            else:
                sim_to_selected = 0.0
            score = lambda_diversity * sim_to_query - (1 - lambda_diversity) * sim_to_selected
            if score > best_score:
                best, best_score = idx, score
        selected.append(best)
        candidates.remove(best)
    return selected


def retrieve(query: str):
    query_emb = embed_texts([query])[0]
    # coarse retrieval
    D, I = st.session_state.faiss_index.search(query_emb.reshape(1, -1), CANDIDATES_K)
    pool = [idx for idx in I[0] if idx < len(st.session_state.chunks)]
    
    # Debug information
    st.session_state.debug_info = {
        "total_chunks": len(st.session_state.chunks),
        "retrieved_initial": len(pool),
        "distances": D[0].tolist()[:5]  # Show top 5 distances
    }
    
    # Make sure we have at least some context by loosening the filter if needed
    if len(pool) > 0:
        # similarity filter (only if we have enough contexts)
        filtered_pool = [idx for idx in pool if D[0][list(I[0]).index(idx)] < (1 - SIM_THRESHOLD)]
        if len(filtered_pool) >= 2:  # Only use filtered pool if we have enough results
            pool = filtered_pool
    
    # If pool is empty after filtering, use the top 3 chunks regardless
    if not pool and len(st.session_state.chunks) > 0:
        pool = [I[0][i] for i in range(min(3, len(I[0]))) if I[0][i] < len(st.session_state.chunks)]
    
    # If still empty (unlikely), use the first chunk as fallback
    if not pool and len(st.session_state.chunks) > 0:
        pool = [0]  # Use the first chunk as a last resort
        
    # MMR rerank if we have a pool
    if pool:
        # Check if we have enough elements for MMR
        if len(pool) >= 2:
            selected = mmr(query_emb, st.session_state.embeddings[pool], min(TOP_K, len(pool)))
            final_idxs = [pool[i] for i in selected]
        else:
            final_idxs = pool
        return [(st.session_state.chunks[i], i) for i in final_idxs]
    
    # No chunks found (should never happen if docs were indexed)
    return []


def rag_prompt(question: str, contexts: List[Tuple[str, int]]):
    if not contexts:
        # Return a prompt asking the model to indicate no information is available
        return [
            {"role": "system", "content": "Vous êtes un assistant expert qui répond en français."},
            {"role": "user", "content": f"Je n'ai pas d'information sur ce sujet dans ma base documentaire. Merci de répondre à l'utilisateur que vous n'avez pas d'information sur : {question}"}
        ]
    
    ctx_block = "\n\n".join(f"[{idx+1}] {text}" for idx, (text, _) in enumerate(contexts))
    system = (
        "Vous êtes un assistant expert. Répondez en français en utilisant UNIQUEMENT les informations fournies "
        "dans les contextes numérotés ci-dessous. N'inventez pas d'information. Référencez vos réponses avec les "
        "numéros de contexte [n]. Si l'information n'est pas disponible dans les contextes, dites simplement "
        "'Je ne trouve pas cette information dans les documents fournis.'"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"CONTEXTES:\n{ctx_block}\n\nQUESTION: {question}\n\nRépondez uniquement à partir des contextes fournis:"}
    ]


async def stream_answer(messages):
    client = st.session_state.async_client
    try:
        stream = await client.chat(
            model=MODEL_NAME, 
            messages=messages, 
            stream=True, 
            options={"temperature": temp, "num_predict": max_tokens}
        )
        async for chunk in stream:
            yield chunk["message"]["content"]
    except Exception:
        # fallback sync
        resp = ollama.chat(
            model=MODEL_NAME, 
            messages=messages, 
            stream=False, 
            options={"temperature": temp, "num_predict": max_tokens}
        )
        yield resp["message"]["content"]

# -----------------------------------------------------------------------------
# Session initialisation -------------------------------------------------------
if "async_client" not in st.session_state:
    st.session_state.async_client = AsyncClient()

if "faiss_index" not in st.session_state and files:
    with st.spinner("🔢 Indexation des documents…"):
        uploaded_docs = []
        for uf in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name
            uploaded_docs.append((uf.name, extract_pdf_text(tmp_path)))
            os.unlink(tmp_path)
        
        # Show preprocessing info
        st.info(f"📄 Préparation de {len(uploaded_docs)} document(s)...")
        
        idx, chunks, emb, meta = build_index(uploaded_docs)
        st.session_state.update(
            faiss_index=idx, chunks=chunks, embeddings=emb, meta=meta, messages=[]
        )

# -----------------------------------------------------------------------------
# Chat interface ---------------------------------------------------------------
if "faiss_index" not in st.session_state:
    st.info("👈 Uploadez d'abord des PDF pour activer le chat.")
    st.stop()

# Show loaded documents indicator
if files:
    st.success(f"✅ {len(files)} document(s) chargé(s) et indexé(s)")
    if "chunks" in st.session_state:
        st.caption(f"Total de {len(st.session_state.chunks)} fragments extraits")
        
        # Add chunk stats to the UI
        if "chunk_stats" in st.session_state:
            stats = st.session_state.chunk_stats
            with st.expander("📊 Statistiques des fragments"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nombre", stats["count"])
                col2.metric("Taille moyenne", f"{stats['avg_length']:.0f} chars")
                col3.metric("Min", stats["min_length"])
                col4.metric("Max", stats["max_length"])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Posez votre question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    contexts = retrieve(user_input)
    
    # Debug information about retrieval
    if "debug_info" in st.session_state:
        with st.expander("🔍 Diagnostic de récupération", expanded=False):
            st.write(f"Total de fragments disponibles: {st.session_state.debug_info['total_chunks']}")
            st.write(f"Fragments initialement récupérés: {st.session_state.debug_info['retrieved_initial']}")
            st.write(f"Distances des premiers résultats: {st.session_state.debug_info['distances']}")
            st.write(f"Fragments finalement utilisés: {len(contexts)}")
    
    messages = rag_prompt(user_input, contexts)

    placeholder = st.chat_message("assistant").empty()

    async def _runner():
        collected = ""
        async for token in stream_answer(messages):
            collected += token
            placeholder.markdown(collected)
        st.session_state.messages.append({"role": "assistant", "content": collected})

    asyncio.run(_runner())

    with st.expander("📝 Contextes utilisés pour la réponse"):
        if contexts:
            for idx, (chunk, chunk_idx) in enumerate(contexts):
                meta = st.session_state.meta[chunk_idx]
                st.markdown(f"**[{idx+1}] {meta['file']} – chunk {meta['chunk_id']}**")
                st.text_area(f"Contenu du fragment {idx+1}", chunk, height=150)
        else:
            st.warning("⚠️ Aucun contexte pertinent n'a été trouvé pour cette question.")

st.write("---")
st.caption("© 2025 – RAG powered by EY (TST tonio) • Streamlit v2")
