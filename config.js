// Configuration for Vertex AI RAG Chat Interface
// ⚠️ WARNING: These credentials are exposed in the browser.
// Only use for demos/prototypes. Set up API key restrictions in GCP Console.

const CONFIG = {
    // Your GCP Project ID
    PROJECT_ID: "astute-sign-476118-i9",

    // Vertex AI location/region
    LOCATION: "us-east4",

    // RAG Corpus name
    CORPUS_NAME: "projects/astute-sign-476118-i9/locations/us-east4/ragCorpora/1866742045545070592",

    // Gemini model to use
    MODEL_NAME: "gemini-2.5-flash",

    // RAG retrieval configuration
    RAG_CONFIG: {
        top_k: 1000,
        vector_distance_threshold: 0.4
    },

    // System prompt for the AI
    SYSTEM_PROMPT: `You are an AI assistant with access to information about Georgia Tech faculty research and you can assume all questions are about Georgia Tech professors so don't mention the university name unless necessary.
When answering questions about professors:
- If asked, provide comprehensive information about their research areas, notable works
- Cite relevant publications and research contributions
- If information is not available in the retrieved context, say so clearly`,

    // API endpoint (you'll need to create a backend proxy for production)
    // For now, we'll use the Vertex AI REST API directly
    API_ENDPOINT: `https://${this.LOCATION || "us-east4"}-aiplatform.googleapis.com/v1/projects/${this.PROJECT_ID || "astute-sign-476118-i9"}/locations/${this.LOCATION || "us-east4"}/publishers/google/models/${this.MODEL_NAME || "gemini-2.5-flash"}:generateContent`,

    // ⚠️ IMPORTANT: You need to set up an API key in GCP Console
    // 1. Go to https://console.cloud.google.com/apis/credentials
    // 2. Create an API Key
    // 3. Restrict it to:
    //    - Vertex AI API
    //    - HTTP referrers (your GitHub Pages URL)
    // 4. Add it here (or better: use a backend proxy)
    API_KEY: "YOUR_API_KEY_HERE"  // Replace with your actual API key
};
