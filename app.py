#!/usr/bin/env python3
"""
CQC Chat Interface - ChatGPT Style
Clean chat interface for CQC content
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    url: str
    title: str
    content: str
    embedding: Optional[np.ndarray] = None

class VectorDatabase:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.documents: List[Document] = []
        self.index = None
        self.embedding_dim = None
        self.metadata = {}
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def load(self, path: str) -> bool:
        """Load the vector database from disk"""
        try:
            if not os.path.exists(path):
                return False
            
            # Load metadata
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load FAISS index
            index_path = os.path.join(path, "faiss_index.bin")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                self.embedding_dim = self.index.d
            
            # Load documents
            doc_path = os.path.join(path, "documents.parquet")
            if os.path.exists(doc_path):
                df = pd.read_parquet(doc_path)
                self.documents = []
                
                for _, row in df.iterrows():
                    doc = Document(
                        url=row['url'],
                        title=row['title'],
                        content=row['content']
                    )
                    self.documents.append(doc)
                
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False

class CQCRAGChat:
    def __init__(self, groq_api_key: str, model: str = "openai/gpt-oss-20b"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.vector_db = VectorDatabase()
        
    def load_database(self, db_path: str) -> bool:
        """Load the vector database"""
        return self.vector_db.load(db_path)
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search documents and return formatted results"""
        results = self.vector_db.search(query, k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'title': doc.title,
                'url': doc.url,
                'content': doc.content,
                'score': score,
                'content_preview': doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            })
        
        return formatted_results
    
    def generate_response(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """Generate response using RAG"""
        # Search for relevant documents
        search_results = self.search_documents(query, k=5)
        
        if not search_results:
            return {
                "answer": "I don't have information to answer this question based on the CQC website content. Please try rephrasing your question or ask about topics related to care quality, inspections, or regulations.",
                "sources": []
            }
        
        # Build context from search results
        context_parts = []
        sources = []
        
        for result in search_results:
            # Limit content length for context
            content = result['content'][:1000] + "..." if len(result['content']) > 1000 else result['content']
            
            context_parts.append(f"Document: {result['title']}\nURL: {result['url']}\nContent: {content}")
            sources.append({
                "title": result['title'],
                "url": result['url']
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build conversation history for context
        conversation_context = ""
        if chat_history:
            recent_history = chat_history[-3:]  # Last 3 exchanges
            for chat in recent_history:
                conversation_context += f"Human: {chat['human']}\nAssistant: {chat['assistant']}\n\n"
        
        # Create the prompt
        system_prompt = """You are a helpful assistant specializing in Care Quality Commission (CQC) information. You answer questions based on the provided CQC website content.

Guidelines:
- Use only the information from the provided documents to answer questions
- Be accurate and cite specific information when possible
- If the documents don't contain enough information, say so clearly
- Be conversational and helpful
- Reference previous conversation context when relevant
- Focus on care quality, inspections, regulations, and guidance topics"""

        user_prompt = f"""Based on the CQC website content below, please answer the user's question.

Website Content:
{context}

Previous Conversation:
{conversation_context}

Current Question: {query}

Please provide a comprehensive answer based on the CQC website content. Be specific and helpful."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
                top_p=0.9
            )
            
            return {
                "answer": response.choices[0].message.content,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": f"Sorry, I encountered an error while generating the response. Please try again.",
                "sources": sources
            }

def main():
    # Page configuration
    st.set_page_config(
        page_title="CQC Chat",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for ChatGPT-like appearance
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .chat-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-header h1 {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #2d3748;
    }
    
    .example-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .example-button {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.875rem;
        color: #4a5568;
    }
    
    .example-button:hover {
        border-color: #cbd5e0;
        background: #f7fafc;
    }
    
    .sources-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid #4299e1;
    }
    
    .source-item {
        margin-bottom: 0.5rem;
    }
    
    .source-item:last-child {
        margin-bottom: 0;
    }
    
    .source-link {
        color: #4299e1;
        text-decoration: none;
        font-weight: 500;
    }
    
    .source-link:hover {
        text-decoration: underline;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat input styling */
    .stChatInput > div {
        border-radius: 24px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
        st.stop()
    
    # Initialize session state
    if 'chat_system' not in st.session_state:
        st.session_state.chat_system = CQCRAGChat(groq_api_key)
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False
    
    # Load database silently
    if not st.session_state.database_loaded:
        db_path = "./vector_db"
        if st.session_state.chat_system.load_database(db_path):
            st.session_state.database_loaded = True
        else:
            st.error("❌ Database not found. Please run the crawler first.")
            st.stop()
    
    # Header
    st.markdown("""
    <div class="chat-header">
        <h1>🏥 CQC Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with example questions
    with st.sidebar:
        st.header("💡 Example Questions")
        
        example_questions = [
            "What is the CQC?",
            "How does CQC inspection work?",
            "What are the fundamental standards?",
            "How to prepare for a CQC inspection?",
            "What are CQC ratings?",
            "CQC registration process",
            "What is the Duty of Candour?",
            "CQC emergency support framework"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"sidebar_{question}", use_container_width=True):
                st.session_state.current_question = question
                st.rerun()
        
        if st.session_state.chat_history:
            st.divider()
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    # Display chat history
    for chat in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.write(chat['human'])
        
        # Assistant message
        with st.chat_message("assistant"):
            st.write(chat['assistant'])
            
            # Show sources in a clean format
            if chat.get('sources'):
                st.markdown("""
                <div class="sources-container">
                    <strong>📚 Sources:</strong><br>
                """, unsafe_allow_html=True)
                
                for i, source in enumerate(chat['sources'], 1):
                    st.markdown(f"""
                    <div class="source-item">
                        {i}. <a href="{source['url']}" target="_blank" class="source-link">{source['title']}</a>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about CQC...")
    
    # Handle example question click
    if hasattr(st.session_state, 'current_question'):
        user_input = st.session_state.current_question
        delattr(st.session_state, 'current_question')
    
    # Process user input
    if user_input:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chat_system.generate_response(
                    user_input, 
                    st.session_state.chat_history
                )
            
            # Display response
            st.write(result['answer'])
            
            # Display sources
            if result['sources']:
                st.markdown("""
                <div class="sources-container">
                    <strong>📚 Sources:</strong><br>
                """, unsafe_allow_html=True)
                
                for i, source in enumerate(result['sources'], 1):
                    st.markdown(f"""
                    <div class="source-item">
                        {i}. <a href="{source['url']}" target="_blank" class="source-link">{source['title']}</a>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Save to chat history
        st.session_state.chat_history.append({
            'human': user_input,
            'assistant': result['answer'],
            'sources': result['sources'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit chat history
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]
        
        st.rerun()

if __name__ == "__main__":
    main()