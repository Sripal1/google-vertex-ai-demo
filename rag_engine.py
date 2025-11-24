from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai

PROJECT_ID = "astute-sign-476118-i9"
display_name = "test_corpus"
paths = ["https://drive.google.com/drive/folders/1kk6JuGoYmMV9U90j48uZLkiC1HzQHLp3"]  # Supports Google Cloud Storage and Google Drive Links

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-east4")

CREATE_NEW_CORPUS = False

if CREATE_NEW_CORPUS:
    print("Creating new RAG Corpus...")
    # Create RagCorpus
    # Configure embedding model, for example "text-embedding-005".
    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )

    rag_corpus = rag.create_corpus(
        display_name=display_name,
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=embedding_model_config
        ),
    )

    # Import Files to the RagCorpus
    print("Importing files (this takes several minutes)...")
    rag.import_files(
        rag_corpus.name,
        paths,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=1024,
                chunk_overlap=200,
            ),
        ),
        max_embedding_requests_per_min=1000,  # Optional
    )
    print(f"Corpus created! Save this name: {rag_corpus.name}")
else:

    print("Using existing RAG Corpus...")
    # New corpus with fixed data (all 177 professors verified)
    corpus_name = "projects/371824203937/locations/us-east4/ragCorpora/3458764513820540928"
    rag_corpus = rag.get_corpus(corpus_name)


rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=50,
    filter=rag.Filter(vector_distance_threshold=0.4),
)

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

system_prompt = """You are an AI assistant with access to information about Georgia Tech faculty research and you can assume all questions are about Georgia Tech professors so don't mention the university name unless necessary.
When answering questions about professors:
- If asked, provide comprehensive information about their research areas, notable works
- Cite relevant publications and research contributions
- If information is not available in the retrieved context, say so clearly"""

rag_model = GenerativeModel(
    model_name="gemini-2.5-flash",
    tools=[rag_retrieval_tool],
    system_instruction=system_prompt
)

response = rag_model.generate_content("tell me about Zsolt Kira's research areas and notable works")
print(response.text)

# Display citations
print("\n" + "="*80)
print("CITATIONS")
print("="*80)

if hasattr(response, 'candidates') and response.candidates:
    for candidate in response.candidates:
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            metadata = candidate.grounding_metadata

            # Display grounding supports (which text segments are backed by which sources)
            if hasattr(metadata, 'grounding_supports'):
                print("\n\nGrounding Details:")
                for support in metadata.grounding_supports:
                    if hasattr(support, 'segment'):
                        segment_text = support.segment.text if hasattr(support.segment, 'text') else 'N/A'
                        chunk_indices = support.grounding_chunk_indices if hasattr(support, 'grounding_chunk_indices') else []
                        confidence = support.confidence_scores if hasattr(support, 'confidence_scores') else 'N/A'

                        print(f"\n  Statement: \"{segment_text}\"")
                        print(f"  Source(s): {[idx+1 for idx in chunk_indices]}")
                        print(f"  Confidence: {confidence:.2%}" if isinstance(confidence, (int, float)) else f"  Confidence: {confidence}")