import streamlit as st

def load_css():
    """Apply custom CSS for better styling of the chatbot interface"""
    st.markdown("""
    <style>
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Chat messages */
        .stChatMessage {
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            transition: transform 0.2s;
        }
        
        .stChatMessage:hover {
            transform: translateY(-2px);
        }
        
        /* User message */
        div[data-testid="stChatMessageContent"]:has(.user-message) {
            background-color: #EBF5FB;
            border-radius: 0.5rem;
        }
        
        /* Assistant message */
        div[data-testid="stChatMessageContent"]:has(.assistant-message) {
            background-color: #F8F9F9;
            border-radius: 0.5rem;
        }
        
        /* Buttons */
        .stButton button {
            border-radius: 2rem;
            font-weight: 600;
            padding: 0.25rem 1rem;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Sidebar */
        .css-1d391kg, .css-163ttbj {
            background-color: #F8F9F9;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #2C3E50;
            font-weight: 700;
        }
        
        /* Welcome message container */
        .welcome-container {
            background-color: #F5F5F5;
            border-left: 5px solid #3498DB;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #3498DB;
        }
        
        /* Code blocks */
        code {
            padding: 0.2em 0.4em;
            border-radius: 3px;
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        pre {
            padding: 1em;
            border-radius: 5px;
            background-color: #f6f8fa;
            overflow-x: auto;
        }
    </style>
    """, unsafe_allow_html=True)
