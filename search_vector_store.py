import os
import json
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreSearcher:
    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Initialize the VectorStoreSearcher with Gemini embedding model.
        """
        self.model_name = model_name
        self.index = None
        self.documents = []
        self.dimension = 768  # Dimension for text-embedding-004

    def load_vector_store(self, store_id: str):
        """
        Load vector store from disk.
        """
        try:
            index_path = f'vector_stores/{store_id}.index'
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Vector store index not found: {index_path}")
            
            self.index = faiss.read_index(index_path)
            if self.index.d != self.dimension:
                raise ValueError(f"Index dimension ({self.index.d}) does not match model dimension ({self.dimension})")
            
            docs_path = f'vector_stores/{store_id}_documents.json'
            if not os.path.exists(docs_path):
                raise FileNotFoundError(f"Vector store documents not found: {docs_path}")
            
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
                
            logger.info(f"Successfully loaded vector store with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using Gemini API.
        """
        try:
            result = genai.embed_content(
                model=f"models/{self.model_name}",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'])
        except Exception as e:
            if "Quota" in str(e) or "429" in str(e):
                logger.warning("Gemini API quota exceeded. Falling back to text-based search.")
                return None
            else:
                logger.error(f"Error getting embedding: {str(e)}")
                raise

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search the vector store with a query and optional filters.
        """
        if not self.documents:
            print("No documents loaded in the vector store.")
            return []

        try:
            # Preprocess query
            processed_query = self.preprocess_query(query)
            
            # Exact Matching
            exact_matches = []
            for doc in self.documents:
                if processed_query in doc['text'].lower() or (
                    'promotion' in doc['metadata'] and processed_query in str(doc['metadata']['promotion']).lower()
                ):
                    exact_matches.append({
                        'document': doc,
                        'score': 1.0
                    })
            
            if exact_matches:
                if filters:
                    exact_matches = [doc for doc in exact_matches if self._apply_filters(doc['document'], filters)]
                return exact_matches[:top_k]

            # Semantic Search
            query_embedding = self.get_embedding(processed_query)
            if query_embedding is None:
                # Fallback to text-based search
                return self.text_based_search(processed_query, filters, top_k, min_score)

            # Vector Search
            distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    if filters and not self._apply_filters(doc, filters):
                        continue
                    score = float(1 / (1 + distances[0][i]))
                    if score >= min_score:
                        results.append({
                            'document': doc,
                            'score': score
                        })
            
            # Sort results by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Fallback to text-based search if no results
            if not results:
                return self.text_based_search(processed_query, filters, top_k, min_score)
            
            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    def print_search_results(self, results: List[Dict[str, Any]]):
        """
        Print search results in a formatted way.
        """
        if not results:
            print("\nNo matching phones found.")
            return
            
        print(f"\n=== Found {len(results)} Matching Phones ===\n")
        
        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result['score']
            
            print(f"Result {i} (Score: {score:.2f}):")
            print(f"Text: {doc['text']}")
            print("\nMetadata:")
            for key, value in doc['metadata'].items():
                print(f"  {key}: {value}")
            print("\n" + "="*50 + "\n")

def search_vector_store(vectorstoreid: str, query: str, filters: Optional[Dict[str, Any]] = None, max_num_result: int = 20) -> List[Dict[str, Any]]:
    """
    Simplified function to search vector store with a single query.
    """
    try:
        searcher = VectorStoreSearcher()
        searcher.load_vector_store(vectorstoreid)
        results = searcher.search(query, filters, top_k=max_num_result)
        return results
    except Exception as e:
        logger.error(f"Error in search_vector_store: {str(e)}")
        return []

def main():
    searcher = VectorStoreSearcher()
    store_id = "phone_vector_store_50_gemini"
    
    try:
        searcher.load_vector_store(store_id)
        query = 'Giảm 2 triệu'
        results = searcher.search(query, filters=None, top_k=None)
        searcher.print_search_results(results)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()