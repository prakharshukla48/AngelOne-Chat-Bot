import os
import pickle
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
from constants import URL

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.documents = []
        self.embeddings = None
        # Threshold for relevance - higher means stricter filtering
        self.relevance_threshold = 1.2  
        
    def create_embeddings(self, documents):
        """Create embeddings for documents"""
        texts = [doc.page_content for doc in documents]
        print(f"Creating embeddings for {len(texts)} documents...")
        
        # Create embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and embeddings
        self.documents = documents
        self.embeddings = embeddings
        
        print(f"Created vector store with {len(documents)} documents")
    
    def _is_valid_query(self, query: str) -> bool:
        """Check if query contains meaningful words"""
        # Remove special characters and split into words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        
        # Filter out pure gibberish (repeated letters/syllables)
        meaningful_words = []
        for word in words:
            # Skip words with too many repeated characters
            if len(set(word)) >= 2 or word in ['i', 'a', 'is', 'it', 'to', 'do', 'go', 'no']:
                meaningful_words.append(word)
        
        # Need at least one meaningful word
        return len(meaningful_words) > 0
        
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar documents with relevance filtering"""
        if self.index is None:
            return []
        
        # Check if query is meaningful
        if not self._is_valid_query(query):
            print(f"Query rejected: '{query}' appears to be gibberish")
            return []
        
        # Create embedding for query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                # Only include results that are reasonably relevant
                if distance <= self.relevance_threshold:
                    results.append((self.documents[idx].page_content, float(distance)))
                else:
                    print(f"Result {i+1} filtered out: distance {distance:.3f} > threshold {self.relevance_threshold}")
        
        # If no relevant results found, return empty list
        if not results:
            print(f"No relevant results found for query: '{query}'")
        
        return results
    
    def save(self, filepath="vector_store.pkl"):
        """Save vector store to file"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'index': faiss.serialize_index(self.index) if self.index is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Vector store saved to {filepath}")
    
    def load(self, filepath="vector_store.pkl"):
        """Load vector store from file"""
        if not os.path.exists(filepath):
            print(f"Vector store file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data.get('documents', [])
            self.embeddings = data.get('embeddings', None)
            
            # Fix: Check if index data exists and is not None
            index_data = data.get('index', None)
            if index_data is not None:
                self.index = faiss.deserialize_index(index_data)
            else:
                self.index = None
            
            print(f"Vector store loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

if __name__ == "__main__":
    # Test the vector store
    from ingest import DocumentIngestor
    
    # Load documents
    ingestor = DocumentIngestor()
    all_file_docs = ingestor.load_all_documents()
    web_docs = ingestor.scrape_webpage(URL)
    all_docs = all_file_docs + web_docs
    chunks = ingestor.process_documents(all_docs)
    
    # Create vector store
    vs = VectorStore()
    vs.create_embeddings(chunks)
    vs.save()
    
    # Test search
    results = vs.search("How to open an account?", k=3)
    for i, (text, score) in enumerate(results):
        print(f"Result {i+1} (Score: {score:.4f}):")
        print(text[:200] + "...")
        print("-" * 50)
