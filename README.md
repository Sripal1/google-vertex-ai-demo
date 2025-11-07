# Georgia Tech Faculty Research Chat Interface

A simple Streamlit-based chat interface for querying information about Georgia Tech faculty research using Vertex AI RAG.

## Features

- Interactive chat interface
- Real-time responses using Gemini 2.5 Flash
- Citation tracking with grounding information
- Conversation history
- Clean, demo-ready UI

## Setup

### Prerequisites

- Python 3.8 or higher
- Google Cloud Project with Vertex AI enabled
- Proper authentication set up for Google Cloud

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Authenticate with Google Cloud (if not already done):
```bash
gcloud auth application-default login
```

### Running the App

Simply run:
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Configuration

The app is configured to use:
- **Project ID**: `astute-sign-476118-i9`
- **Location**: `us-east4`
- **Corpus**: Pre-existing RAG corpus with Georgia Tech faculty research data
- **Model**: `gemini-2.5-flash`

To modify these settings, edit the constants at the top of `streamlit_app.py`.

## Usage

1. Type your question in the chat input box
2. Press Enter or click Send
3. View the AI-generated response
4. Click "View Citations" to see grounding information and sources

### Example Questions

- "Tell me about Zsolt Kira's research areas and notable works"
- "What are Professor X's main research contributions?"
- "Who works on machine learning at Georgia Tech?"

## Notes

- The chat history is stored in session state and will reset when you refresh the page
- Use the "Clear Chat History" button in the sidebar to start a fresh conversation
- Citations show which parts of the response are backed by retrieved documents
