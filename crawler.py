#!/usr/bin/env python3
"""
CQC Website RAG Crawler - Command Line Version
Automatically crawls https://www.everythingcqc.com and builds vector database
"""

import os
import re
import time
import logging
import argparse
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dataclasses import dataclass
from datetime import datetime
import json
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    url: str
    title: str
    content: str
    embedding: Optional[np.ndarray] = None

class WebsiteCrawler:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """Extract URLs from sitemap XML"""
        try:
            logger.info(f"Fetching sitemap from: {sitemap_url}")
            response = self.session.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            urls = []
            
            # Try different XML structures
            # Standard sitemap format
            for elem in root:
                for child in elem:
                    if child.tag.endswith('loc'):
                        urls.append(child.text.strip())
            
            # If no URLs found, try with namespaces
            if not urls:
                for elem in root.iter():
                    if elem.tag.endswith('}loc') or elem.tag == 'loc':
                        if elem.text:
                            urls.append(elem.text.strip())
            
            # Remove duplicates and filter valid URLs
            urls = list(set(url for url in urls if url and url.startswith('http')))
            
            logger.info(f"Found {len(urls)} URLs in sitemap")
            return urls
            
        except Exception as e:
            logger.error(f"Error parsing sitemap: {e}")
            return []
    
    def extract_text_from_url(self, url: str) -> Optional[Document]:
        """Extract text content from a single URL"""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                element.decompose()
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Try different content selectors in order of preference
            content_selectors = [
                'main',
                'article', 
                '.content',
                '#content',
                '.main-content',
                '.post-content',
                '.entry-content',
                '[role="main"]'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = " ".join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    break
            
            # Fallback to body content
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            # Clean up text
            content = re.sub(r'\s+', ' ', content).strip()
            content = re.sub(r'\n+', '\n', content)
            
            # Filter out pages with minimal content
            if len(content) < 200:
                logger.debug(f"Skipping {url} - insufficient content ({len(content)} chars)")
                return None
                
            # Filter out navigation/menu pages
            if any(word in content.lower() for word in ['sitemap', 'page not found', '404', 'error']):
                logger.debug(f"Skipping {url} - appears to be navigation/error page")
                return None
            
            logger.debug(f"Successfully extracted content from {url} ({len(content)} chars)")
            return Document(url=url, title=title, content=content)
            
        except requests.RequestException as e:
            logger.warning(f"Request error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {e}")
            return None

class VectorDatabase:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing vector database with model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.documents: List[Document] = []
        self.index = None
        self.embedding_dim = None
    
    def add_documents(self, documents: List[Document], batch_size: int = 32):
        """Add documents to the vector database"""
        if not documents:
            return
            
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        
        texts = [doc.content for doc in documents]
        
        # Generate embeddings in batches with progress bar
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Store embeddings in documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        self.documents.extend(documents)
        
        # Initialize or update FAISS index
        if self.index is None:
            self.embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"Created FAISS index with dimension {self.embedding_dim}")
        
        # Normalize and add to index
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        logger.info(f"Added {len(documents)} documents to vector database")
    
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
    
    def save(self, path: str):
        """Save the vector database to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))
            logger.info(f"Saved FAISS index to {path}")
        
        # Save documents
        if self.documents:
            doc_data = []
            for doc in self.documents:
                doc_data.append({
                    'url': doc.url,
                    'title': doc.title,
                    'content': doc.content,
                    'content_length': len(doc.content)
                })
            
            df = pd.DataFrame(doc_data)
            df.to_parquet(os.path.join(path, "documents.parquet"))
            logger.info(f"Saved {len(doc_data)} documents to {path}")
        
        # Save metadata
        metadata = {
            'total_documents': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'embedding_model': self.model.get_sentence_embedding_dimension(),
            'created_at': datetime.now().isoformat(),
            'total_content_length': sum(len(doc.content) for doc in self.documents)
        }
        
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Vector database saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load the vector database from disk"""
        try:
            if not os.path.exists(path):
                logger.warning(f"Database path {path} does not exist")
                return False
            
            # Load metadata
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loading database with {metadata.get('total_documents', 0)} documents")
            
            # Load FAISS index
            index_path = os.path.join(path, "faiss_index.bin")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                self.embedding_dim = self.index.d
                logger.info(f"Loaded FAISS index with dimension {self.embedding_dim}")
            
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
                
                logger.info(f"Loaded {len(self.documents)} documents")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False

class RAGSystem:
    def __init__(self, groq_api_key: str, model: str = "llama3-groq-70b-8192-tool-use-preview"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.vector_db = VectorDatabase()
        self.crawler = WebsiteCrawler()
    
    def crawl_and_build_database(self, sitemap_url: str, max_pages: Optional[int] = None, output_dir: str = "./vector_db"):
        """Crawl website and build vector database"""
        
        # Get all URLs from sitemap
        logger.info("Starting website crawl...")
        all_urls = self.crawler.get_sitemap_urls(sitemap_url)
        
        if not all_urls:
            raise ValueError("No URLs found in sitemap")
        
        # Limit pages if specified
        if max_pages:
            urls = all_urls[:max_pages]
            logger.info(f"Limited to {max_pages} pages out of {len(all_urls)} total")
        else:
            urls = all_urls
        
        logger.info(f"Crawling {len(urls)} pages...")
        
        # Crawl pages with progress bar
        documents = []
        failed_urls = []
        
        for url in tqdm(urls, desc="Crawling pages"):
            doc = self.crawler.extract_text_from_url(url)
            if doc:
                documents.append(doc)
            else:
                failed_urls.append(url)
        
        if not documents:
            raise ValueError("No documents were crawled successfully")
        
        logger.info(f"Successfully crawled {len(documents)} pages")
        if failed_urls:
            logger.info(f"Failed to crawl {len(failed_urls)} pages")
        
        # Add to vector database
        logger.info("Building vector database...")
        self.vector_db.add_documents(documents)
        
        # Save database
        logger.info(f"Saving database to {output_dir}...")
        self.vector_db.save(output_dir)
        
        # Print summary
        total_content = sum(len(doc.content) for doc in documents)
        avg_content = total_content // len(documents)
        
        print("\n" + "="*50)
        print("CRAWLING COMPLETE!")
        print("="*50)
        print(f"✅ Successfully crawled: {len(documents)} pages")
        print(f"❌ Failed to crawl: {len(failed_urls)} pages")
        print(f"📄 Total content: {total_content:,} characters")
        print(f"📊 Average per page: {avg_content:,} characters")
        print(f"💾 Database saved to: {output_dir}")
        print("="*50)
        
        return len(documents)
    
    def test_search(self, query: str, k: int = 3):
        """Test the search functionality"""
        results = self.vector_db.search(query, k)
        
        print(f"\n🔍 Search Results for: '{query}'")
        print("-" * 50)
        
        if not results:
            print("No results found.")
            return
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Title: {doc.title}")
            print(f"   URL: {doc.url}")
            print(f"   Content preview: {doc.content[:200]}...")

def main():
    parser = argparse.ArgumentParser(description="CQC Website RAG Crawler")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to crawl")
    parser.add_argument("--output-dir", default="./vector_db", help="Output directory for vector database")
    parser.add_argument("--test-query", help="Test query to run after crawling")
    parser.add_argument("--load-existing", action="store_true", help="Load existing database and test search")
    
    args = parser.parse_args()
    
    # Check for Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables")
        print("Please add your Groq API key to your .env file")
        return 1
    
    # CQC sitemap URL
    sitemap_url = "https://www.everythingcqc.com/index.php?option=com_jmap&view=sitemap&format=xml"
    
    try:
        # Initialize RAG system
        rag = RAGSystem(groq_api_key)
        
        if args.load_existing:
            # Load existing database
            if rag.vector_db.load(args.output_dir):
                print(f"✅ Loaded existing database from {args.output_dir}")
            else:
                print(f"❌ Failed to load database from {args.output_dir}")
                return 1
        else:
            # Crawl and build database
            doc_count = rag.crawl_and_build_database(
                sitemap_url=sitemap_url,
                max_pages=args.max_pages,
                output_dir=args.output_dir
            )
        
        # Test search if query provided
        if args.test_query:
            rag.test_search(args.test_query)
        
        print(f"\n🎉 RAG system ready! Database location: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())