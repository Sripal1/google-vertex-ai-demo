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

REMEMBER: Every factual statement must be wrapped in <CIT> tags directly in your response!
"""

rag_model = GenerativeModel(
    model_name="gemini-2.5-flash",
    tools=[rag_retrieval_tool],
    system_instruction=system_prompt
)

# Generate content with inline citations
from vertexai.generative_models import GenerationConfig
import re
from typing import List, Dict, Any

################################################################################
# STEP 1: SPLIT CHUNKS INTO SENTENCES AND CREATE SENTENCE MAP
################################################################################

def split_chunk_into_sentences(chunk_text: str, chunk_id: int) -> List[Dict[str, Any]]:
    """
    Split a chunk into sentences and track their position (sentence_id, start_char, end_char).
    Returns a list of sentence dictionaries.
    """
    # Simple sentence splitting (you can use more sophisticated methods like spaCy)
    sentence_pattern = r'[.!?]+\s+'
    sentences = []
    current_pos = 0
    sentence_id = 1

    # Split by sentence boundaries
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
    """
    Create a mapping from chunk_id to list of sentences with their metadata.
    """
    sentence_map = {}
    for chunk_id, chunk_text in enumerate(chunks):
        sentence_map[chunk_id] = split_chunk_into_sentences(chunk_text, chunk_id)
    return sentence_map

################################################################################
# STEP 2: RETRIEVE CHUNKS FROM VERTEX AI RAG RESPONSE
################################################################################

def get_retrieved_chunks(response_obj) -> List[str]:
    """
    Extract retrieved chunks from Vertex AI RAG response metadata.
    """
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

################################################################################
# STEP 3: PARSE THE LLM RESPONSE TO EXTRACT CITATIONS
################################################################################

def parse_response_with_citations(response_text: str) -> Dict[str, Any]:
    """
    Parse <CIT chunk_id='N' sentences='X-Y'>snippet</CIT> tags.
    Returns:
    {
      "text": <final answer with tags removed>,
      "citations": [
        {
          "chunk_id": int,
          "sentences_range": "X-Y",
          "answer_snippet": snippet text from answer,
          "answer_snippet_start": char position in answer,
          "answer_snippet_end": char position in answer
        }
      ]
    }
    """
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

################################################################################
# STEP 4: MATCH CITED SENTENCES AND FIND CHARACTER RANGES
################################################################################

def gather_sentence_data_for_citations(
    parsed_response: Dict[str, Any],
    sentence_map: Dict[int, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    For each citation, look up the actual sentence text from the sentence_map
    and record the character ranges in the original chunk.
    """
    for citation in parsed_response["citations"]:
        chunk_id = citation["chunk_id"]
        sent_range = citation["sentences_range"]

        try:
            start_sent, end_sent = map(int, sent_range.split("-"))
        except:
            start_sent, end_sent = 1, 1

        # Get sentences for this chunk
        sentences_for_chunk = sentence_map.get(chunk_id, [])

        # Filter to the cited sentence range
        relevant_sentences = [
            s for s in sentences_for_chunk
            if start_sent <= s["sentence_id"] <= end_sent
        ]

        if relevant_sentences:
            # Combine the sentence texts
            combined_text = " ".join(s["text"] for s in relevant_sentences)
            chunk_start_char = relevant_sentences[0]["start_char"]
            chunk_end_char = relevant_sentences[-1]["end_char"]
        else:
            combined_text = ""
            chunk_start_char = -1
            chunk_end_char = -1

        # Add this info to the citation
        citation["chunk_sentences_text"] = combined_text
        citation["chunk_sentences_start"] = chunk_start_char
        citation["chunk_sentences_end"] = chunk_end_char

    return parsed_response

################################################################################
# MAIN EXECUTION
################################################################################

response = rag_model.generate_content(
    "tell me about Alex Endert's research areas and his most important project",
    generation_config=GenerationConfig(
        temperature=0.0,  # More deterministic for better grounding
    )
)

# Display the raw response with inline citations
print("\n" + "="*80)
print("RESPONSE WITH INLINE CITATIONS (Raw)")
print("="*80)
print(response.text)
print("="*80)

# Step 1: Get retrieved chunks from Vertex AI response
retrieved_chunks = get_retrieved_chunks(response)

if not retrieved_chunks:
    print("\n⚠️  No chunks retrieved from Vertex AI RAG")
else:
    print(f"\n✓ Retrieved {len(retrieved_chunks)} chunks from Vertex AI RAG")

    # Step 2: Create sentence map
    sentence_map = create_sentence_map(retrieved_chunks)

    # Step 3: Parse citations from response
    parsed_response = parse_response_with_citations(response.text)

    # Step 4: Map citations to source sentences
    parsed_response = gather_sentence_data_for_citations(parsed_response, sentence_map)

    # Display results
    print("\n" + "="*80)
    print("FINAL ANSWER (Clean Text)")
    print("="*80)
    print(parsed_response["text"])
    print("="*80)

    if not parsed_response["citations"]:
        print("\n⚠️  No citations found. Model may not have followed citation format.")
    else:
        print(f"\n✓ Found {len(parsed_response['citations'])} citations\n")
        print("="*80)
        print("CITATION DETAILS")
        print("="*80)

        for idx, citation in enumerate(parsed_response["citations"], 1):
            print(f"\n[Citation {idx}]")
            print(f"📝 Answer snippet: \"{citation['answer_snippet']}\"")
            print(f"📎 Chunk ID: {citation['chunk_id']}")
            print(f"📄 Sentence range: {citation['sentences_range']}")
            print(f"📍 Position in answer: chars {citation['answer_snippet_start']}-{citation['answer_snippet_end']}")
            print(f"\n📚 Source sentences text:")
            print(f"   \"{citation['chunk_sentences_text']}\"")
            print(f"📍 Position in chunk: chars {citation['chunk_sentences_start']}-{citation['chunk_sentences_end']}")
            print("-" * 80)

    # # Display all chunks with sentence breakdown
    # print("\n" + "="*80)
    # print("ALL RETRIEVED CHUNKS WITH SENTENCE BREAKDOWN")
    # print("="*80)
    # for chunk_id, chunk_text in enumerate(retrieved_chunks):
    #     print(f"\n[Chunk {chunk_id}]")
    #     print(f"Full text: {chunk_text[:200]}..." if len(chunk_text) > 200 else f"Full text: {chunk_text}")
    #     print(f"\nSentences:")
    #     for sent in sentence_map[chunk_id]:
    #         print(f"  [{sent['sentence_id']}] chars {sent['start_char']}-{sent['end_char']}: {sent['text'][:100]}...")
    #     print("-" * 80)