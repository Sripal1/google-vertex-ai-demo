import streamlit as st
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
from google.oauth2 import service_account

# Page configuration
st.set_page_config(
    page_title="Georgia Tech Faculty Research Chat",
    page_icon="🎓",
    layout="wide"
)

# Initialize Vertex AI
PROJECT_ID = "astute-sign-476118-i9"
CORPUS_NAME = "projects/astute-sign-476118-i9/locations/us-east4/ragCorpora/1866742045545070592"

@st.cache_resource
def initialize_rag_model():
    """Initialize the RAG model (cached to avoid reinitializing)"""
    # For Streamlit Cloud: load from secrets
    try:
        if "gcp_service_account" in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            vertexai.init(project=PROJECT_ID, location="us-east4", credentials=credentials)
        else:
            # Local development
            vertexai.init(project=PROJECT_ID, location="us-east4")
    except FileNotFoundError:
        # No secrets file - local development
        vertexai.init(project=PROJECT_ID, location="us-east4")

    # Get existing corpus
    rag_corpus = rag.get_corpus(CORPUS_NAME)

    # Configure RAG retrieval
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=50,
        filter=rag.Filter(vector_distance_threshold=0.4),
    )

    # Create retrieval tool
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=rag_corpus.name,
                    )
                ],
                rag_retrieval_config=rag_retrieval_config,
            ),
        )
    )

    # System prompt
    system_prompt = """You are an AI assistant with access to information about Georgia Tech faculty research and you can assume all questions are about Georgia Tech professors so don't mention the university name unless necessary.
When answering questions about professors:
- If asked, provide comprehensive information about their research areas, notable works
- Cite relevant publications and research contributions
- If information is not available in the retrieved context, say so clearly"""

    # Initialize model
    model = GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=[rag_retrieval_tool],
        system_instruction=system_prompt
    )

    return model

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    with st.spinner("Initializing RAG model..."):
        st.session_state.model = initialize_rag_model()

if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.model.start_chat()

# Header
col1, col2 = st.columns([6, 1])
with col1:
    st.title("🎓 Georgia Tech CoC Researchers")
    st.markdown("Ask questions about Georgia Tech professors, their research areas, and publications.")
with col2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_session = st.session_state.model.start_chat()
        st.rerun()

st.divider()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about a professor or research area..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(prompt)
            response_text = response.text

            # Display response
            st.markdown(response_text)

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text
    })
