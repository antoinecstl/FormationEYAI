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
MODEL_NAME = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"
DOC_TOP_K = 3
CHUNK_TOP_K = 5
CANDIDATES_K = 50  # Augmenté pour plus de diversité
NEIGHBORS = 1
LAMBDA_DIVERSITY = 0.4  # Plus de diversité vs similarité pure
SIM_THRESHOLD = 0.4     # Plus permissif (était 0.25)
TEMPERATURE = 0.2
MAX_TOKENS = 2048

# HNSW optimisé pour embeddings 768D
HNSW_M = 48            # Connectivité (16-64 optimal pour 768D)
HNSW_EF_CONSTRUCTION = 200  # Qualité construction
HNSW_EF_SEARCH = 100   # Vitesse/qualité recherche
FLAT_THRESHOLD = 10000  # Seuil pour passer en IndexFlatIP

# Optimisations avancées
NEIGHBOR_SIM_RATIO = 0.7  # Seuil pour voisins (70% du seuil principal)
DOC_FILTER_MULTIPLIER = 5  # Facteur de sur-échantillonnage pour filtrage post-hoc

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

        # Normalisation des embeddings documents pour cosine similarity
        doc_emb_matrix = np.vstack(doc_embs).astype("float32")
        doc_emb_matrix = doc_emb_matrix / np.linalg.norm(doc_emb_matrix, axis=1, keepdims=True)
        
        self.doc_index = faiss.IndexFlatIP(doc_emb_matrix.shape[1])
        self.doc_index.add(doc_emb_matrix)

        # Normalisation des embeddings chunks pour cohérence métrique
        self.chunk_emb = np.vstack(chunk_embs_list).astype("float32")
        self.chunk_emb = self.chunk_emb / np.linalg.norm(self.chunk_emb, axis=1, keepdims=True)
        
        # Choix automatique d'index selon la taille
        if len(self.chunk_emb) < FLAT_THRESHOLD:
            # IndexFlatIP pour petites collections (plus précis)
            self.chunk_index = faiss.IndexFlatIP(self.chunk_emb.shape[1])
        else:
            # HNSW optimisé avec métrique INNER_PRODUCT cohérente
            self.chunk_index = faiss.IndexHNSWFlat(self.chunk_emb.shape[1], 
                                                  HNSW_M, 
                                                  faiss.METRIC_INNER_PRODUCT)
            self.chunk_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            self.chunk_index.hnsw.efSearch = HNSW_EF_SEARCH
        
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
        # Normalisation de la query pour cohérence métrique
        q_emb = embed_texts([query])[0]
        q_emb = q_emb / np.linalg.norm(q_emb)
        
        # 1. Sélection des documents pertinents
        _, I_doc = self.doc_index.search(q_emb[None, :], DOC_TOP_K)
        allowed_docs = set(I_doc[0])

        # 2. Recherche directe dans l'index HNSW/Flat (SANS recréation)
        if len(allowed_docs) == len(self.doc_meta):
            # Tous les docs → recherche globale
            D, I = self.chunk_index.search(q_emb[None, :], CANDIDATES_K)
            pool_indices = I[0]
            pool_scores = D[0]
        else:
            # Filtrage par document + recherche large puis filtrage
            # Recherche plus large pour compenser le filtrage post-hoc
            search_k = min(CANDIDATES_K * 5, len(self.chunk_meta))  # Augmenté de 3→5
            D, I = self.chunk_index.search(q_emb[None, :], search_k)
            
            # Filtrage des chunks par document autorisé
            filtered_pairs = [(idx, score) for idx, score in zip(I[0], D[0]) 
                            if self.chunk_meta[idx]["doc_id"] in allowed_docs]
            
            if not filtered_pairs:
                # Fallback: prendre le premier chunk du premier doc autorisé
                fallback_chunks = [i for i, m in enumerate(self.chunk_meta) 
                                 if m["doc_id"] in allowed_docs]
                pool_indices = [fallback_chunks[0]] if fallback_chunks else [0]
                pool_scores = [0.5]  # Score neutre pour IP normalisé
            else:
                pool_indices, pool_scores = zip(*filtered_pairs[:CANDIDATES_K])
        
        # 3. Filtrage par seuil de similarité (uniformisé pour IP)
        # Avec vecteurs normalisés: IP élevé = similaire pour tous les index
        valid_pairs = [(idx, score) for idx, score in zip(pool_indices, pool_scores) 
                      if score >= SIM_THRESHOLD]
        
        if not valid_pairs:
            # Fallback: prendre le meilleur candidat
            valid_pairs = [(pool_indices[0], pool_scores[0])]
        
        pool = [idx for idx, _ in valid_pairs]
        
        # 4. MMR pour diversifier
        if len(pool) > CHUNK_TOP_K:
            cand_emb = self.chunk_emb[pool]
            selected_mmr = self._mmr(q_emb, cand_emb, CHUNK_TOP_K)
            selected = [pool[i] for i in selected_mmr]
        else:
            selected = pool
        
        # 5. Expansion contextuelle intelligente (chunks voisins avec seuil)
        expanded = set(selected)  # Commencer avec les chunks sélectionnés
        for idx in selected:
            for offset in range(-NEIGHBORS, NEIGHBORS + 1):
                if offset == 0:  # Skip le chunk central (déjà dans selected)
                    continue
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.chunk_meta):
                    # Vérifier que le voisin est du même document
                    if self.chunk_meta[neighbor_idx]["doc_id"] == self.chunk_meta[idx]["doc_id"]:
                        # Seuil de similarité pour éviter les voisins non-pertinents
                        neighbor_sim = float(q_emb @ self.chunk_emb[neighbor_idx])
                        if neighbor_sim >= SIM_THRESHOLD * 0.7:  # 70% du seuil principal
                            expanded.add(neighbor_idx)
        
        # Trier par score de similarité (ordre décroissant pour IP)
        final_with_scores = []
        for idx in expanded:
            score = float(q_emb @ self.chunk_emb[idx])
            final_with_scores.append((idx, score))
        
        final_with_scores.sort(key=lambda x: x[1], reverse=True)
        final = [idx for idx, _ in final_with_scores[:CHUNK_TOP_K]]
        
        # 6. Extraction des contextes et résumé
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
    st.session_state.processing = True

    rag: RagIndex = st.session_state.rag  # type: ignore
    
    # Récupération des contextes avec feedback visuel
    with st.spinner("🔍 Recherche des passages pertinents..."):
        contexts, indices, summary = rag.retrieve(query)
    
    # Construction du prompt
    prompt = build_prompt(query, contexts, summary, st.session_state.messages)

    placeholder = st.empty()
    collected_parts: List[str] = []

    # Génération en streaming
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
