# CQC RAG Chat Assistant 🏥

A Retrieval-Augmented Generation (RAG) chat system for Care Quality Commission (CQC) content, powered by Groq's OpenAI GPT OSS 20B model.

## Features

- 🤖 **AI-Powered Chat**: Ask questions about CQC regulations, inspections, and guidance
- 🔍 **Semantic Search**: Uses vector embeddings for accurate content retrieval
- 📚 **Source Citations**: All answers include relevant source links
- 🎯 **CQC-Focused**: Trained specifically on CQC website content
- ⚡ **Fast Responses**: Powered by Groq's high-speed inference

## Setup

### Prerequisites
- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/keys))

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd cqc-rag-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

5. **Crawl CQC website** (builds vector database)
```bash
python crawler.py --max-pages 50
```

6. **Run the chat app**
```bash
streamlit run app.py
```

## Usage

### Crawling
```bash
# Crawl specific number of pages
python crawler.py --max-pages 50

# Crawl all pages
python crawler.py

# Test search after crawling
python crawler.py --max-pages 30 --test-query "What is CQC?"
```

### Chat Interface
- Open http://localhost:8501
- Use example questions from sidebar
- Ask about CQC inspections, regulations, ratings, etc.
- View source citations for all answers

## Example Questions

- "What is the CQC?"
- "How does CQC inspection work?"
- "What are the fundamental standards?"
- "How to prepare for a CQC inspection?"
- "What are CQC ratings?"
- "CQC registration process"

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq OpenAI GPT OSS 20B
- **Embeddings**: SentenceTransformers
- **Vector DB**: FAISS
- **Web Scraping**: BeautifulSoup + Requests

## Project Structure

```
cqc-rag-system/
├── app.py              # Streamlit chat interface
├── crawler.py          # Website crawler
├── requirements.txt    # Python dependencies
├── .env.example       # Environment template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Configuration

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Model Options
You can change the model in `app.py`:
- `openai/gpt-oss-20b` (default)
- `llama3-groq-70b-8192-tool-use-preview`
- Other Groq-supported models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- Open a GitHub issue
- Check Groq documentation: https://docs.groq.com