// Chat Interface Logic
// Handles user interactions, API calls, and message rendering

class ChatInterface {
    constructor() {
        this.messages = [];
        this.messagesContainer = document.getElementById('messages');
        this.userInput = document.getElementById('userInput');
        this.sendButton = document.getElementById('sendButton');
        this.errorMessage = document.getElementById('errorMessage');

        this.initializeEventListeners();
        this.configureMarked();
    }

    initializeEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Enter key to send (Shift+Enter for new line)
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = this.userInput.scrollHeight + 'px';
        });

        // Check API key on load
        this.checkApiKey();
    }

    configureMarked() {
        // Configure marked.js for markdown rendering
        marked.setOptions({
            breaks: true,
            gfm: true,
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.error('Highlighting error:', err);
                    }
                }
                return hljs.highlightAuto(code).value;
            }
        });
    }

    checkApiKey() {
        if (!CONFIG.API_KEY || CONFIG.API_KEY === 'YOUR_API_KEY_HERE') {
            this.showError(
                '⚠️ API Key not configured. Please add your Vertex AI API key to config.js. ' +
                'See README.md for instructions.'
            );
            this.sendButton.disabled = true;
            this.userInput.disabled = true;
        }
    }

    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message) return;

        // Clear input and disable while processing
        this.userInput.value = '';
        this.userInput.style.height = 'auto';
        this.setLoading(true);
        this.hideError();

        // Add user message
        this.addMessage(message, 'user');

        try {
            // Call Vertex AI API
            const response = await this.callVertexAI(message);

            // Add bot response
            this.addMessage(response.text, 'bot', response.citations);
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Error: ${error.message}`);
            this.addMessage(
                'Sorry, I encountered an error processing your request. Please try again.',
                'bot'
            );
        } finally {
            this.setLoading(false);
        }
    }

    async callVertexAI(userMessage) {
        // Note: This uses the Vertex AI REST API with generateContent endpoint
        // For production, you should use a backend proxy to keep credentials secure

        const url = `https://${CONFIG.LOCATION}-aiplatform.googleapis.com/v1/projects/${CONFIG.PROJECT_ID}/locations/${CONFIG.LOCATION}/publishers/google/models/${CONFIG.MODEL_NAME}:generateContent`;

        const requestBody = {
            contents: [
                {
                    role: "user",
                    parts: [
                        { text: userMessage }
                    ]
                }
            ],
            systemInstruction: {
                parts: [
                    { text: CONFIG.SYSTEM_PROMPT }
                ]
            },
            tools: [
                {
                    retrieval: {
                        vertexRagStore: {
                            ragResources: [
                                {
                                    ragCorpus: CONFIG.CORPUS_NAME
                                }
                            ],
                            ragRetrievalConfig: {
                                topK: CONFIG.RAG_CONFIG.top_k,
                                filter: {
                                    vectorDistanceThreshold: CONFIG.RAG_CONFIG.vector_distance_threshold
                                }
                            }
                        }
                    }
                }
            ],
            generationConfig: {
                temperature: 0.7,
                topP: 0.95,
                maxOutputTokens: 2048
            }
        };

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${CONFIG.API_KEY}`
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(
                errorData.error?.message ||
                `API request failed with status ${response.status}`
            );
        }

        const data = await response.json();

        // Extract text and citations from response
        return this.parseResponse(data);
    }

    parseResponse(data) {
        const result = {
            text: '',
            citations: []
        };

        if (!data.candidates || data.candidates.length === 0) {
            result.text = 'No response generated.';
            return result;
        }

        const candidate = data.candidates[0];

        // Extract main text
        if (candidate.content && candidate.content.parts) {
            result.text = candidate.content.parts
                .map(part => part.text || '')
                .join(' ');
        }

        // Extract grounding metadata (citations)
        if (candidate.groundingMetadata) {
            const metadata = candidate.groundingMetadata;

            if (metadata.groundingSupports) {
                result.citations = metadata.groundingSupports.map(support => ({
                    segment: support.segment?.text || '',
                    sources: support.groundingChunkIndices || [],
                    confidence: support.confidenceScores?.[0] || null
                }));
            }
        }

        return result;
    }

    addMessage(text, sender, citations = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (sender === 'bot') {
            // Render markdown for bot messages
            const sanitizedHtml = DOMPurify.sanitize(marked.parse(text));
            contentDiv.innerHTML = sanitizedHtml;

            // Add citations if available
            if (citations && citations.length > 0) {
                const citationsDiv = this.createCitationsElement(citations);
                contentDiv.appendChild(citationsDiv);
            }
        } else {
            // Plain text for user messages
            contentDiv.textContent = text;
        }

        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;

        // Store message
        this.messages.push({ text, sender, citations, timestamp: new Date() });
    }

    createCitationsElement(citations) {
        const citationsDiv = document.createElement('div');
        citationsDiv.className = 'citations';

        const title = document.createElement('div');
        title.className = 'citations-title';
        title.textContent = '📚 Sources:';
        citationsDiv.appendChild(title);

        citations.forEach((citation, index) => {
            if (citation.segment) {
                const citationItem = document.createElement('div');
                citationItem.className = 'citation-item';

                let citationText = `"${citation.segment}"`;

                if (citation.sources && citation.sources.length > 0) {
                    citationText += ` [Sources: ${citation.sources.map(s => s + 1).join(', ')}]`;
                }

                if (citation.confidence !== null) {
                    citationText += ` (Confidence: ${(citation.confidence * 100).toFixed(1)}%)`;
                }

                citationItem.textContent = citationText;
                citationsDiv.appendChild(citationItem);
            }
        });

        return citationsDiv;
    }

    setLoading(isLoading) {
        this.sendButton.disabled = isLoading;
        this.userInput.disabled = isLoading;

        if (isLoading) {
            this.sendButton.innerHTML = '<span class="loading">Thinking</span>';
        } else {
            this.sendButton.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
            `;
            this.userInput.focus();
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.classList.add('show');
    }

    hideError() {
        this.errorMessage.classList.remove('show');
    }
}

// Initialize chat interface when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});
