# Georgia Tech Faculty Research Chat Interface

A simple, browser-based chat interface for querying Georgia Tech faculty research using Vertex AI RAG (Retrieval Augmented Generation).

## Features

- 🎓 Clean, modern chat interface
- 💬 Real-time conversation with AI assistant
- 📚 RAG-powered responses with source citations
- 🎨 Markdown and code syntax highlighting
- 📱 Responsive design for mobile and desktop

## Demo

This interface can be hosted on GitHub Pages for easy access and sharing.

## ⚠️ Security Warning

**This is a demo/prototype implementation only!**

This implementation exposes API credentials in the browser, which is **not secure for production use**. Only use this for:
- Personal demos
- Internal prototypes
- Educational purposes

### For Production Use

For production deployments, you should:
1. **Create a backend API proxy** (Cloud Functions, Cloud Run, etc.) that:
   - Securely stores credentials on the server
   - Handles authentication and authorization
   - Proxies requests to Vertex AI
2. **Implement proper authentication** for your users
3. **Set up rate limiting** and usage quotas

## Setup Instructions

### Prerequisites

1. Google Cloud Project with Vertex AI enabled
2. RAG Corpus created and populated (see `rag_engine.py`)
3. API key or OAuth credentials for Vertex AI

### Step 1: Get Your API Key

You have two options for authentication:

#### Option A: API Key (Simpler, but requires restrictions)

1. Go to [Google Cloud Console - API Credentials](https://console.cloud.google.com/apis/credentials)
2. Click **Create Credentials** → **API Key**
3. **IMPORTANT: Restrict the API key:**
   - Click on the created key to edit it
   - Under "API restrictions":
     - Select "Restrict key"
     - Enable only: **Vertex AI API**
   - Under "Website restrictions":
     - Add your GitHub Pages URL (e.g., `https://yourusername.github.io/*`)
     - Add `http://localhost:*` for local testing
4. Copy the API key

#### Option B: OAuth 2.0 (More secure, requires code changes)

See [Google Cloud Authentication docs](https://cloud.google.com/docs/authentication) for implementing OAuth flow.

### Step 2: Configure the Application

1. Open `config.js`
2. Update the following values:
   ```javascript
   API_KEY: "YOUR_API_KEY_HERE"  // Replace with your actual API key
   ```

   If you need to change other settings:
   ```javascript
   PROJECT_ID: "your-project-id"
   LOCATION: "us-east4"  // or your region
   CORPUS_NAME: "projects/.../ragCorpora/..."  // Your corpus name
   MODEL_NAME: "gemini-2.5-flash"  // or another model
   ```

### Step 3: Test Locally

1. Open `index.html` in a web browser directly, or
2. Use a local web server (recommended):
   ```bash
   # Using Python 3
   python3 -m http.server 8000

   # Using Node.js
   npx http-server

   # Using PHP
   php -S localhost:8000
   ```
3. Visit `http://localhost:8000` in your browser

### Step 4: Deploy to GitHub Pages

1. **Create a new GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Georgia Tech faculty chat interface"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Enable GitHub Pages:**
   - Go to your repository on GitHub
   - Click **Settings** → **Pages**
   - Under "Source", select `main` branch and `/ (root)` folder
   - Click **Save**

3. **Update API key restrictions:**
   - Go back to [Google Cloud Console - API Credentials](https://console.cloud.google.com/apis/credentials)
   - Edit your API key
   - Under "Website restrictions", add your GitHub Pages URL:
     - `https://YOUR_USERNAME.github.io/*`

4. **Access your site:**
   - Your site will be available at: `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`
   - It may take a few minutes for GitHub Pages to deploy

## File Structure

```
vertex-ai/
├── index.html          # Main HTML structure
├── style.css           # Styling and layout
├── chat.js             # Chat logic and API integration
├── config.js           # Configuration (API keys, project settings)
├── rag_engine.py       # Backend RAG corpus setup (Python)
└── README.md           # This file
```

## Customization

### Changing the System Prompt

Edit the `SYSTEM_PROMPT` in `config.js` to change how the AI responds:

```javascript
SYSTEM_PROMPT: `Your custom instructions here...`
```

### Modifying RAG Settings

Adjust retrieval behavior in `config.js`:

```javascript
RAG_CONFIG: {
    top_k: 1000,                      // Number of chunks to retrieve
    vector_distance_threshold: 0.4    // Minimum similarity threshold
}
```

### Styling

Edit `style.css` to customize:
- Colors (see CSS variables at the top)
- Layout and spacing
- Fonts and typography
- Responsive breakpoints

## Troubleshooting

### "API Key not configured" error
- Make sure you've updated `config.js` with your actual API key
- Refresh the page after saving changes

### "API request failed with status 403"
- Check that your API key is valid
- Verify API restrictions allow Vertex AI API
- Ensure website restrictions include your domain

### "API request failed with status 401"
- Your API key may be invalid or expired
- Try creating a new API key

### CORS errors
- Make sure you're using HTTPS (GitHub Pages) or localhost
- Check that website restrictions in Google Cloud Console are set correctly
- The Vertex AI API should support CORS for browser requests

### No citations showing
- Citations depend on the RAG system finding relevant sources
- Try adjusting `vector_distance_threshold` in config.js
- Verify your RAG corpus is properly populated

## Cost Considerations

This application makes API calls to Vertex AI, which may incur costs:
- Gemini API calls (per request)
- RAG retrieval operations
- Embedding operations

Monitor your usage in [Google Cloud Console - Billing](https://console.cloud.google.com/billing)

Set up budget alerts to avoid unexpected charges.

## Development

### Adding New Features

To add features like:
- Conversation history export
- Clear chat button
- Multiple conversation threads
- User preferences

Edit `chat.js` and add corresponding UI elements in `index.html`.

### Using a Different Model

To use a different Gemini model, update `config.js`:

```javascript
MODEL_NAME: "gemini-1.5-pro"  // or another model
```

See [Vertex AI Models](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models) for available options.

## License

This is a demo project. Modify and use as needed for your purposes.

## Support

For issues with:
- **Vertex AI API**: See [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs)
- **RAG Setup**: Refer to `rag_engine.py` and [RAG documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/rag-api)
- **This interface**: Check the troubleshooting section above

## Acknowledgments

Built with:
- [Vertex AI](https://cloud.google.com/vertex-ai) - Google's AI platform
- [Marked.js](https://marked.js.org/) - Markdown parsing
- [Highlight.js](https://highlightjs.org/) - Code syntax highlighting
- [DOMPurify](https://github.com/cure53/DOMPurify) - XSS protection
