import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM 
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
import requests
import time
import json
from datetime import datetime
from styles import load_css

vectoring_model = "nomic-embed-text:latest" 
OLLAMA_BASE_URL = "http://localhost:11434"  

# Callback handler for streaming responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")
        
    def on_llm_end(self, response, **kwargs):
        self.container.markdown(self.text)

# Fonction pour vérifier si Ollama est disponible
def is_ollama_running():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False

# Functions for PDF processing
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model=vectoring_model)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = OllamaLLM(                                          
        model=st.session_state.reasoning_model,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        base_url=OLLAMA_BASE_URL,
        streaming=True,
        callbacks=[st.session_state.stream_handler]
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
def export_chat_history():
    if not st.session_state.chat_history:
        return None
    
    export_data = []
    for msg in st.session_state.chat_history:
        export_data.append({
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return json.dumps(export_data, indent=2)

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "PDF AI Assistant with RAG capability powered by Ollama"
    }
)

# Apply CSS from styles.py
load_css()

# Initialize session state
if "app_loaded" not in st.session_state:
    st.session_state.app_loaded = True
    st.session_state.reasoning_model = "llama3.2:3b"
    st.session_state.temperature = 0.7
    st.session_state.max_tokens = 1024
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.processing = False
    st.session_state.vectorstore = None
    # Initialize stream handler with a default empty placeholder
    st.session_state.stream_handler = StreamHandler(st.empty())
    st.rerun()

# App Sidebar
with st.sidebar:
    st.title("PDF AI Assistant 🤖")
    
    # Model Configuration
    st.header("Model Configuration")
    
    st.session_state.reasoning_model = st.radio(
        "Choose a model:",
        ["llama3.2:3b", "gemma3:12b"],
        index=0,
        help="Select the model to use for reasoning"
    )
    
    st.session_state.temperature = st.slider(
        "Temperature:", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.7, 
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    st.session_state.max_tokens = st.slider(
        "Max Output Tokens:", 
        min_value=256, 
        max_value=4096, 
        value=1024, 
        step=128,
        help="Maximum number of tokens in the model response"
    )
    
    # PDF Upload Section
    st.header("Upload Documents")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
    
    if st.button("Process PDFs", use_container_width=True, type="primary"):
        if not is_ollama_running():
            st.error("⚠️ Cannot connect to Ollama. Make sure the server is running with 'ollama serve' command.")
        elif not pdf_docs:
            st.error("⚠️ Please upload PDF files first.")
        else:
            with st.spinner("Processing PDFs with Ollama..."):
                try:
                    progress_bar = st.progress(0)
                    
                    # Extract text
                    st.info("Extracting text from PDFs...")
                    raw_text = get_pdf_text(pdf_docs)
                    progress_bar.progress(25)
                    
                    # Split text
                    st.info(f"Text extracted: {len(raw_text)} characters. Splitting into chunks...")
                    text_chunks = get_text_chunks(raw_text)
                    progress_bar.progress(50)
                    
                    # Create embeddings
                    st.info(f"Text split into {len(text_chunks)} chunks. Creating embeddings with Ollama...")
                    vectorstore = get_vectorstore(text_chunks)
                    progress_bar.progress(75)
                    
                    # Store vectorstore in session state for later reuse
                    st.session_state.vectorstore = vectorstore
                    
                    # Setup conversation chain
                    st.info("Setting up the conversation chain...")
                    placeholder = st.empty()
                    st.session_state.stream_handler = StreamHandler(placeholder)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    progress_bar.progress(100)
                    
                    st.success("PDFs Processed Successfully!")
                    time.sleep(1)
                    progress_bar.empty()
                    
                except Exception as e:
                    import traceback
                    st.error(f"Error processing PDFs: {str(e)}")
                    st.code(traceback.format_exc(), language="python")
    
    # File listing
    if pdf_docs:
        st.header("Uploaded Files")
        for doc in pdf_docs:
            st.markdown(f"📄 **{doc.name}**")
    
    # Export functionality
    if st.session_state.chat_history:
        st.header("Export Chat")
        if st.download_button(
            label="Download Chat History",
            data=export_chat_history(),
            file_name="chat_export.json",
            mime="application/json",
            use_container_width=True
        ):
            st.success("Chat history downloaded!")

# Main Content Area
st.title("AI PDF Assistant 💬")

# Welcome message when no conversation
if not st.session_state.conversation:
    st.markdown("""
    <div class="welcome-box">
        <h3>👋 Welcome to the AI PDF Assistant!</h3>
        <p>To get started:</p>
        <ol>
            <li>Upload one or more PDF documents using the sidebar</li>
            <li>Click "Process PDFs" to analyze the documents</li>
            <li>Ask questions about your documents in the chat input below</li>
        </ol>
        <p>The assistant will analyze your documents and answer questions based on their content.</p>
    </div>
    """, unsafe_allow_html=True)

# Chat interface controls
col1, col2 = st.columns([5, 1])
with col1:
    user_question = st.chat_input(
        "Ask a question about your documents...", 
        disabled=st.session_state.processing or not st.session_state.conversation
    )
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = None
        if st.session_state.conversation:
            # Reset memory but keep the documents
            st.session_state.conversation.memory.clear()
        st.rerun()

# Chat container
chat_container = st.container(height=500)

with chat_container:
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message("user", avatar="👤"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)
    
    # Process new user questions
    if user_question:
        with st.chat_message("user", avatar="👤"):
            st.write(user_question)
        
        with st.chat_message("assistant", avatar="🤖"):
            if st.session_state.conversation:
                st.session_state.processing = True
                try:
                    # Create a placeholder for streaming text inside the assistant message
                    response_placeholder = st.empty()
                    # Update the stream handler to use the new placeholder
                    st.session_state.stream_handler = StreamHandler(response_placeholder)
                    
                    # Since we can't directly access the LLM in ConversationalRetrievalChain,
                    # recreate the conversation chain with the new stream handler
                    if st.session_state.vectorstore is not None:
                        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                    
                    # Process the user question and generate a response
                    handle_userinput(user_question)
                finally:
                    st.session_state.processing = False
            else:
                st.error("Please process PDF documents before asking questions!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Ollama | PDF AI Assistant v1.0")