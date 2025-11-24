import streamlit as st
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool, GenerationConfig
import vertexai
from google.oauth2 import service_account
import re
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Georgia Tech Faculty Research Chat",
    page_icon="🎓",  # Keep page icon for browser tab
    layout="wide"
)

# Initialize Vertex AI
PROJECT_ID = "astute-sign-476118-i9"
CORPUS_NAME = "projects/371824203937/locations/us-east4/ragCorpora/3458764513820540928"

################################################################################
# CITATION PROCESSING FUNCTIONS
################################################################################

def split_chunk_into_sentences(chunk_text: str, chunk_id: int) -> List[Dict[str, Any]]:
    """Split a chunk into sentences and track their position."""
    sentence_pattern = r'[.!?]+\s+'
    sentences = []
    current_pos = 0
    sentence_id = 1

    parts = re.split(sentence_pattern, chunk_text)

    for part in parts:
        if part.strip():
            start_char = chunk_text.find(part, current_pos)
            end_char = start_char + len(part)
            sentences.append({
                "chunk_id": chunk_id,
                "sentence_id": sentence_id,
                "text": part.strip(),
                "start_char": start_char,
                "end_char": end_char
            })
            sentence_id += 1
            current_pos = end_char

    return sentences

def create_sentence_map(chunks: List[str]) -> Dict[int, List[Dict[str, Any]]]:
    """Create a mapping from chunk_id to list of sentences."""
    sentence_map = {}
    for chunk_id, chunk_text in enumerate(chunks):
        sentence_map[chunk_id] = split_chunk_into_sentences(chunk_text, chunk_id)
    return sentence_map

def get_retrieved_chunks(response_obj) -> List[str]:
    """Extract retrieved chunks from Vertex AI RAG response."""
    retrieved_chunks = []
    if hasattr(response_obj, 'candidates') and response_obj.candidates:
        for candidate in response_obj.candidates:
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'retrieved_context'):
                            chunk_text = chunk.retrieved_context.text
                            retrieved_chunks.append(chunk_text)
    return retrieved_chunks

def parse_response_with_citations(response_text: str) -> Dict[str, Any]:
    """Parse <CIT chunk_id='N' sentences='X-Y'>snippet</CIT> tags."""
    pattern = re.compile(
        r"<CIT\s+chunk_id=['\"](\d+)['\"]\s+sentences=['\"]([^'\"]+)['\"]>(.*?)</CIT>",
        re.DOTALL
    )

    final_text = ""
    citations = []
    last_end = 0

    for match in pattern.finditer(response_text):
        # Add text before this citation
        final_text += response_text[last_end:match.start()]

        chunk_id = int(match.group(1))
        sent_range = match.group(2)
        snippet = match.group(3)

        # Record where this snippet appears in the final answer
        start_in_answer = len(final_text)
        final_text += snippet
        end_in_answer = len(final_text)

        citations.append({
            "chunk_id": chunk_id,
            "sentences_range": sent_range,
            "answer_snippet": snippet,
            "answer_snippet_start": start_in_answer,
            "answer_snippet_end": end_in_answer
        })

        last_end = match.end()

    # Add remaining text
    final_text += response_text[last_end:]

    return {
        "text": final_text,
        "citations": citations
    }

def find_relevant_text_in_chunk(chunk_text: str, answer_snippet: str, context_chars: int = 200) -> str:
    """
    Search for the answer snippet or similar text in the chunk and return relevant context.
    Uses multiple strategies to find the most relevant text.
    """
    # Strategy 1: Direct substring match (case-insensitive)
    snippet_lower = answer_snippet.lower().strip()
    chunk_lower = chunk_text.lower()

    idx = chunk_lower.find(snippet_lower)
    if idx != -1:
        # Found exact match - return with surrounding context
        start = max(0, idx - context_chars)
        end = min(len(chunk_text), idx + len(answer_snippet) + context_chars)
        return chunk_text[start:end].strip()

    # Strategy 2: Find overlapping phrases (at least 3 words)
    snippet_words = snippet_lower.split()
    if len(snippet_words) >= 3:
        # Try to find sequences of 3+ consecutive words
        for window_size in range(min(len(snippet_words), 5), 2, -1):
            for i in range(len(snippet_words) - window_size + 1):
                phrase = " ".join(snippet_words[i:i+window_size])
                idx = chunk_lower.find(phrase)
                if idx != -1:
                    # Found partial match
                    start = max(0, idx - context_chars)
                    end = min(len(chunk_text), idx + len(phrase) + context_chars)
                    return chunk_text[start:end].strip()

    # Strategy 3: Keyword-based search (find sentences with most matching keywords)
    # Extract key terms (words longer than 3 chars, excluding common words)
    common_words = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'has', 'are', 'was', 'were'}
    keywords = [w for w in snippet_words if len(w) > 3 and w not in common_words]

    if keywords:
        # Split chunk into sentences and score each by keyword matches
        sentences = re.split(r'[.!?]+\s+', chunk_text)
        best_sentence = None
        best_score = 0
        best_idx = 0

        for idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = sum(1 for kw in keywords if kw in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence
                best_idx = idx

        if best_sentence and best_score > 0:
            # Return best matching sentence with context (prev + current + next sentences)
            start_idx = max(0, best_idx - 1)
            end_idx = min(len(sentences), best_idx + 2)
            context_sentences = sentences[start_idx:end_idx]
            return " ".join(context_sentences).strip()

    # Strategy 4: Fallback - return beginning of chunk (more likely to be relevant than random middle)
    return chunk_text[:400] + ("..." if len(chunk_text) > 400 else "")

def gather_sentence_data_for_citations(
    parsed_response: Dict[str, Any],
    sentence_map: Dict[int, List[Dict[str, Any]]],
    retrieved_chunks: List[str]
) -> Dict[str, Any]:
    """Map citations to actual source sentences."""
    for citation in parsed_response["citations"]:
        chunk_id = citation["chunk_id"]
        sent_range = citation["sentences_range"]
        answer_snippet = citation["answer_snippet"]

        try:
            start_sent, end_sent = map(int, sent_range.split("-"))
        except:
            start_sent, end_sent = 1, 1

        sentences_for_chunk = sentence_map.get(chunk_id, [])
        relevant_sentences = [
            s for s in sentences_for_chunk
            if start_sent <= s["sentence_id"] <= end_sent
        ]

        if relevant_sentences:
            combined_text = " ".join(s["text"] for s in relevant_sentences)
            chunk_start_char = relevant_sentences[0]["start_char"]
            chunk_end_char = relevant_sentences[-1]["end_char"]
        else:
            # Smarter fallback: Search for the answer snippet in the chunk
            if chunk_id < len(retrieved_chunks):
                chunk_text = retrieved_chunks[chunk_id]
                combined_text = find_relevant_text_in_chunk(chunk_text, answer_snippet)
            else:
                combined_text = "Source text not available"
            chunk_start_char = 0
            chunk_end_char = len(combined_text)

        citation["chunk_sentences_text"] = combined_text
        citation["chunk_sentences_start"] = chunk_start_char
        citation["chunk_sentences_end"] = chunk_end_char

    return parsed_response

def format_text_with_citation_numbers(parsed_response: Dict[str, Any]) -> str:
    """Replace citation snippets with numbered citations [1], [2], etc."""
    text = parsed_response["text"]
    citations = parsed_response["citations"]

    # Sort citations by position in reverse order to replace from end to start
    sorted_citations = sorted(enumerate(citations, 1),
                            key=lambda x: x[1]["answer_snippet_start"],
                            reverse=True)

    for idx, citation in sorted_citations:
        snippet = citation["answer_snippet"]
        start = citation["answer_snippet_start"]
        end = citation["answer_snippet_end"]

        # Replace the snippet with snippet + citation number
        text = text[:start] + snippet + f" [{idx}]" + text[end:]

    # Clean up any remaining malformed citation tags
    text = clean_citation_tags(text)

    return text

def clean_citation_tags(text: str) -> str:
    """Remove any remaining <CIT> tags (both opening and closing) from text."""
    # Remove opening tags: <CIT chunk_id='X' sentences='Y-Z'>
    text = re.sub(r"<CIT\s+chunk_id=['\"]?\d+['\"]?\s+sentences=['\"][^'\"]+['\"]>", "", text)
    # Remove closing tags: </CIT>
    text = re.sub(r"</CIT>", "", text)
    # Remove any malformed variations
    text = re.sub(r"<CIT[^>]*>", "", text)
    return text

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

    # System prompt with inline citation instructions
    system_prompt = """You are an AI assistant with access to information about Georgia Tech faculty research and you can assume all questions are about Georgia Tech professors so don't mention the university name unless necessary.

CRITICAL: You MUST use inline citation tags within your response text. This is mandatory and non-negotiable.

CITATION FORMAT - YOU MUST FOLLOW THIS EXACTLY:
- Wrap EVERY factual claim with: <CIT chunk_id='X' sentences='Y-Z'>your claim text here</CIT>
- 'chunk_id' is the index of the retrieved chunk (0, 1, 2, etc.)
- 'sentences' is the sentence range in that chunk (e.g., '1-3', '5-7')
- The text inside <CIT> tags is YOUR ANSWER TEXT, not the source text
- DO NOT create a separate "Sources:" section - use only inline <CIT> tags

CORRECT EXAMPLE:
<CIT chunk_id='0' sentences='1-2'>Professor Kira specializes in machine learning and robotics</CIT>. His work focuses on <CIT chunk_id='0' sentences='5-7'>continual learning and out-of-distribution detection</CIT>. He developed <CIT chunk_id='1' sentences='3-4'>Habitat 2.0 and 3.0 simulation platforms</CIT>.

INCORRECT (DO NOT DO THIS):
Professor Kira specializes in machine learning and robotics. His work focuses on continual learning.

Sources:
[0] sentences 1-2: "text from source"

REMEMBER: Every factual statement must be wrapped in <CIT> tags directly in your response!"""

    # Initialize model
    model = GenerativeModel(
        model_name="gemini-3-pro-preview",
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
    st.title("Georgia Tech CoC Researchers")
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
        # Clean any citation tags from history
        content = message["content"]
        if message["role"] == "assistant":
            content = clean_citation_tags(content)
        st.markdown(content)

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
            response = st.session_state.chat_session.send_message(
                prompt,
                generation_config=GenerationConfig(temperature=0.0)
            )
            response_text = response.text

            # Parse citations
            retrieved_chunks = get_retrieved_chunks(response)

            if retrieved_chunks:
                # Create sentence map
                sentence_map = create_sentence_map(retrieved_chunks)

                # Parse response with citations
                parsed_response = parse_response_with_citations(response_text)

                # Map citations to source sentences
                parsed_response = gather_sentence_data_for_citations(parsed_response, sentence_map, retrieved_chunks)

                if parsed_response["citations"]:
                    # Format text with citation numbers
                    formatted_text = format_text_with_citation_numbers(parsed_response)

                    # Display formatted response with citation numbers
                    st.markdown(formatted_text)

                    # Display citations section
                    with st.expander(f"View Citations ({len(parsed_response['citations'])})", expanded=False):
                        for idx, citation in enumerate(parsed_response["citations"], 1):
                            st.markdown(f"**[{idx}]** {citation['answer_snippet']}")

                            if citation.get('chunk_sentences_text'):
                                with st.container():
                                    st.text_area(
                                        f"Source text:",
                                        citation['chunk_sentences_text'],
                                        height=150,
                                        key=f"citation_{idx}",
                                        disabled=True
                                    )
                            st.divider()

                    # Store formatted text with citations for history
                    display_content = formatted_text
                else:
                    # No citations found, clean up any malformed tags and display
                    cleaned_text = clean_citation_tags(response_text)
                    st.markdown(cleaned_text)
                    display_content = cleaned_text
            else:
                # No chunks retrieved, clean up any malformed tags and display
                cleaned_text = clean_citation_tags(response_text)
                st.markdown(cleaned_text)
                display_content = cleaned_text

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": display_content
    })
