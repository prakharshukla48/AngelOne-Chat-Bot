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
        self.dimension = 384  
        self.index = None
        self.documents = []
        self.embeddings = None
        self.relevance_threshold = 1.8
        
    def create_embeddings(self, documents):
        
        texts = [doc.page_content for doc in documents]
        print(f"Creating embeddings for {len(texts)} documents...")
        
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        
        self.documents = documents
        self.embeddings = embeddings
        
        print(f"Created vector store with {len(documents)} documents")
    
    def _is_valid_query(self, query: str) -> bool:
        
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        
        
        meaningful_words = []
        for word in words:
            
            if len(set(word)) >= 2 or word in ['i', 'a', 'is', 'it', 'to', 'do', 'go', 'no']:
                meaningful_words.append(word)
        
        
        return len(meaningful_words) > 0
        
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        
        if self.index is None:
            return []
        
        
        if not self._is_valid_query(query):
            print(f"Query rejected: '{query}' appears to be gibberish")
            return []
        
        
        query_embedding = self.model.encode([query])
        
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):

                if distance <= self.relevance_threshold:
                    results.append((self.documents[idx].page_content, float(distance)))
                else:
                    print(f"Result {i+1} filtered out: distance {distance:.3f} > threshold {self.relevance_threshold}")
        
        
        if not results:
            print(f"No relevant results found for query: '{query}'")
        
        return results
    
    def save(self, filepath="vector_store.pkl"):
        
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'index': faiss.serialize_index(self.index) if self.index is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Vector store saved to {filepath}")
    
    def load(self, filepath="vector_store.pkl"):
        
        if not os.path.exists(filepath):
            print(f"Vector store file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data.get('documents', [])
            self.embeddings = data.get('embeddings', None)
            
            
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
    
    from ingest import DocumentIngestor
    
    
    ingestor = DocumentIngestor()
    all_file_docs = ingestor.load_all_documents()
    web_docs = ingestor.scrape_webpage(URL)
    all_docs = all_file_docs + web_docs
    chunks = ingestor.process_documents(all_docs)
    
    
    vs = VectorStore()
    vs.create_embeddings(chunks)
    vs.save()
    

    results = vs.search("How to open an account?", k=3)
    for i, (text, score) in enumerate(results):
        print(f"Result {i+1} (Score: {score:.4f}):")
        print(text[:200] + "...")
        print("-" * 50)
