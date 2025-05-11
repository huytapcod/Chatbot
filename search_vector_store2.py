import os
import json
import numpy as np
import faiss
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, model_name: str = "text-embedding-004", max_workers: int = 4):
        """
        Initialize the VectorStoreSearcher with Gemini embedding model.
        """
        self.model_name = model_name
        self.index = None
        self.documents = []
        self.dimension = 768  # Dimension for text-embedding-004
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

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

    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using Gemini API asynchronously.
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=text,
                    task_type="retrieval_document"
                )
            )
            return np.array(result['embedding'])
        except Exception as e:
            if "Quota" in str(e) or "429" in str(e):
                logger.warning("Gemini API quota exceeded. Falling back to text-based search.")
                return None
            else:
                logger.error(f"Error getting embedding: {str(e)}")
                raise

    async def analyze_with_gemini(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze search results using Gemini for better relevance scoring.
        """
        try:
            if not results:
                return results

            # Prepare context for Gemini
            context = "Analyze these search results and their relevance to the query. Return a JSON with scores (0-1) for each result:\n"
            context += f"Query: {query}\n\nResults:\n"
            
            for i, result in enumerate(results):
                doc = result['document']
                context += f"{i+1}. {doc['metadata']['title']} - {doc['metadata']['description']}\n"

            # Get Gemini's analysis
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.model.generate_content(context)
            )
            
            # Parse Gemini's response and update scores
            try:
                analysis = json.loads(response.text)
                for i, result in enumerate(results):
                    if str(i+1) in analysis:
                        # Combine original score with Gemini's analysis
                        result['score'] = (result['score'] + analysis[str(i+1)]) / 2
            except:
                logger.warning("Could not parse Gemini's analysis, using original scores")

            return sorted(results, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            logger.error(f"Error in Gemini analysis: {str(e)}")
            return results

    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search the vector store with a query and optional filters using async/await.
        """
        if not self.documents:
            print("No documents loaded in the vector store.")
            return []

        try:
            # Preprocess query
            processed_query = query.lower().strip()
            print(f"\nĐang tìm kiếm với từ khóa: {processed_query}")
            
            # Direct text search
            results = []
            for doc in self.documents:
                score = 0.0
                text = doc['text'].lower()
                metadata = doc['metadata']
                
                # Check text field
                if processed_query in text:
                    score = 1.0
                else:
                    # Check individual terms
                    query_terms = processed_query.split()
                    matches = sum(1 for term in query_terms if term in text)
                    if matches > 0:
                        score = matches / len(query_terms)
                
                # Check metadata fields
                if score == 0:
                    for key, value in metadata.items():
                        if value and isinstance(value, (str, int, float)):
                            value_str = str(value).lower()
                            if processed_query in value_str:
                                score = 0.8
                                break
                            else:
                                # Check individual terms in metadata
                                matches = sum(1 for term in query_terms if term in value_str)
                                if matches > 0:
                                    score = matches / len(query_terms) * 0.8
                                    break
                
                if score >= min_score:
                    if filters and not self._apply_filters(doc, filters):
                        continue
                    results.append({
                        'document': doc,
                        'score': score
                    })
            
            if results:
                print(f"Tìm thấy {len(results)} kết quả phù hợp")
                results.sort(key=lambda x: x['score'], reverse=True)
                results = results[:top_k]
                
                # Analyze results with Gemini
                results = await self.analyze_with_gemini(processed_query, results)
                return results

            # If no direct matches, try semantic search
            print("\nĐang tìm kiếm ngữ nghĩa...")
            query_embedding = await self.get_embedding(processed_query)
            if query_embedding is None:
                print("Không thể tạo vector nhúng, chuyển sang tìm kiếm văn bản")
                return await self.text_based_search(processed_query, filters, top_k, min_score)

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
            
            # Sort results by score and analyze with Gemini
            results.sort(key=lambda x: x['score'], reverse=True)
            results = await self.analyze_with_gemini(processed_query, results)
            print(f"Tìm thấy {len(results)} kết quả phù hợp")
            
            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []

    def _apply_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Apply filters to a document.
        Returns True if document passes all filters, False otherwise.
        """
        try:
            for key, value in filters.items():
                if key not in doc['metadata']:
                    return False
                
                if key in ['min_price', 'max_price']:
                    price = doc['metadata']['price']
                    if key == 'min_price' and price < value:
                        return False
                    if key == 'max_price' and price > value:
                        return False
                elif isinstance(value, str):
                    # Case-insensitive string comparison
                    if str(doc['metadata'][key]).lower() != value.lower():
                        return False
                else:
                    # Exact comparison for non-string values
                    if doc['metadata'][key] != value:
                        return False
            return True
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return False

    def print_search_results(self, results: List[Dict[str, Any]]):
        """
        Print search results in a formatted way.
        """
        if not results:
            print("\nKhông tìm thấy sản phẩm phù hợp.")
            return
            
        print(f"\n=== Tìm thấy {len(results)} sản phẩm phù hợp ===\n")
        
        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result['score']
            
            print(f"Kết quả {i} (Độ phù hợp: {score:.2f}):")
            print(f"Sản phẩm: {doc['metadata']['title']}")
            print(f"Giá: {doc['metadata']['price']:,} VND".replace(',', '.'))
            print(f"Màu sắc: {doc['metadata']['color']}")
            print(f"Thông số: {doc['metadata']['specs']}")
            print(f"Khuyến mãi: {doc['metadata']['promotion']}")
            print(f"Mô tả: {doc['metadata']['description']}")
            print("\n" + "="*50 + "\n")

    def _extract_price_range(self, query: str) -> Optional[Dict[str, int]]:
        """
        Extract price range from query text.
        Returns a dictionary with min_price and/or max_price if found.
        """
        try:
            # Convert query to lowercase for easier matching
            query = query.lower()
            
            # Initialize price range
            price_range = {}
            
            # Common price range patterns
            patterns = [
                (r'(\d+(?:\.\d+)?)\s*(?:triệu|tr)\s*(?:đến|tới|-)\s*(\d+(?:\.\d+)?)\s*(?:triệu|tr)', 1000000),
                (r'(\d+(?:\.\d+)?)\s*(?:triệu|tr)\s*(?:trở lên|trở xuống)', 1000000),
                (r'(\d+(?:\.\d+)?)\s*(?:triệu|tr)\s*(?:đồng)?', 1000000),
                (r'(\d+(?:\.\d+)?)\s*(?:k|nghìn)\s*(?:đến|tới|-)\s*(\d+(?:\.\d+)?)\s*(?:k|nghìn)', 1000),
                (r'(\d+(?:\.\d+)?)\s*(?:k|nghìn)\s*(?:trở lên|trở xuống)', 1000),
                (r'(\d+(?:\.\d+)?)\s*(?:k|nghìn)\s*(?:đồng)?', 1000)
            ]
            
            import re
            for pattern, multiplier in patterns:
                matches = re.findall(pattern, query)
                if matches:
                    for match in matches:
                        if len(match) == 2:  # Range pattern
                            min_price = float(match[0]) * multiplier
                            max_price = float(match[1]) * multiplier
                            price_range['min_price'] = int(min_price)
                            price_range['max_price'] = int(max_price)
                        else:  # Single value pattern
                            price = float(match[0]) * multiplier
                            if 'trở lên' in query:
                                price_range['min_price'] = int(price)
                            elif 'trở xuống' in query:
                                price_range['max_price'] = int(price)
                            else:
                                # If no specific direction, use as both min and max
                                price_range['min_price'] = int(price * 0.9)  # 10% below
                                price_range['max_price'] = int(price * 1.1)  # 10% above
            
            return price_range if price_range else None
            
        except Exception as e:
            logger.error(f"Error extracting price range: {str(e)}")
            return None

    def _extract_color(self, query: str) -> Optional[str]:
        """
        Extract color from query text.
        Returns the color if found, None otherwise.
        """
        try:
            # Convert query to lowercase for easier matching
            query = query.lower()
            
            # Common color mappings
            color_mappings = {
                'đen': 'Đen',
                'trắng': 'Trắng',
                'xanh': 'Xanh',
                'xanh dương': 'Xanh dương',
                'xanh lá': 'Xanh lá',
                'đỏ': 'Đỏ',
                'hồng': 'Hồng',
                'tím': 'Tím',
                'vàng': 'Vàng',
                'bạc': 'Bạc',
                'xám': 'Xám',
                'nâu': 'Nâu',
                'cam': 'Cam'
            }
            
            # Check for color mentions
            for color_key, color_value in color_mappings.items():
                if color_key in query:
                    return color_value
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting color: {str(e)}")
            return None

    def search_by_screen_size(self, size: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for phones with specific screen size.
        """
        if not self.documents:
            print("No documents loaded in the vector store.")
            return []

        try:
            results = []
            for doc in self.documents:
                metadata = doc['metadata']
                specs = metadata.get('specs', '').lower()
                
                # Check if screen size is mentioned in specs
                if size in specs:
                    results.append({
                        'document': doc,
                        'score': 1.0
                    })
            
            # Sort by score and limit results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error during screen size search: {str(e)}")
            return []

async def main():
    searcher = VectorStoreSearcher()
    store_id = "phone_vector_store_50_gemini"
    
    try:
        searcher.load_vector_store(store_id)
        screen_size = "4.7 inch"
        print(f"\nĐang tìm kiếm điện thoại màn hình {screen_size}...")
        
        results = searcher.search_by_screen_size(screen_size)
        searcher.print_search_results(results)
            
    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        searcher.executor.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 