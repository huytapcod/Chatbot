import os
import json
import numpy as np
import faiss
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import time
from typing import List, Dict, Any

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Initialize the VectorStoreBuilder with Gemini embedding model.
        """
        self.model_name = model_name
        self.dimension = 768  # Dimension for text-embedding-004
        self.index = None
        self.documents = []
        
    def load_phone_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load phone data from CSV file and prepare it for vector store.
        """
        try:
            df = pd.read_csv(data_path)
            required_columns = ['id', 'title', 'product_specs', 'color', 'price', 'description', 'url', 'product_promotion']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
            documents = []
            for _, row in df.iterrows():
                text = f"{row['title']} - {row['product_specs']} - Color: {row['color']} - Price: {row['price']} VND - Description: {row['description']}"
                doc = {
                    'text': text,
                    'metadata': {
                        'id': row['id'],
                        'title': row['title'],
                        'url': row['url'],
                        'promotion': row['product_promotion'],
                        'specs': row['product_specs'],
                        'color': row['color'],
                        'price': row['price'],
                        'description': row['description']
                    }
                }
                documents.append(doc)
                
            logger.info(f"Successfully loaded {len(documents)} phone documents from {data_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading phone data from {data_path}: {str(e)}")
            raise

    def build_vector_store(self, documents: List[Dict[str, Any]]) -> str:
        """
        Build vector store from documents.
        """
        try:
            texts = [doc.get('text', '') for doc in documents]
            
            # Generate embeddings using Gemini
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store documents
            self.documents = documents
            
            # Generate a unique ID for this vector store
            store_id = f"phone_vector_store_{len(documents)}_gemini"
            
            # Save the index and documents
            self._save_vector_store(store_id)
            
            logger.info(f"Successfully built vector store with ID: {store_id}")
            return store_id
            
        except Exception as e:
            logger.error(f"Error building vector store: {str(e)}")
            raise

    def _save_vector_store(self, store_id: str):
        """
        Save the vector store to disk.
        """
        try:
            os.makedirs('vector_stores', exist_ok=True)
            faiss.write_index(self.index, f'vector_stores/{store_id}.index')
            with open(f'vector_stores/{store_id}_documents.json', 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved vector store with ID: {store_id}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

def main():
    builder = VectorStoreBuilder()
    data_path = "data/phonedatas (3).csv"
    if os.path.exists(data_path):
        documents = builder.load_phone_data(data_path)
        store_id = builder.build_vector_store(documents)
        print(f"Created phone vector store with ID: {store_id}")
    else:
        print(f"Error: File {data_path} not found")

if __name__ == "__main__":
    main()