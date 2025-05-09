import os
import json
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Check for API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhoneChatbot:
    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Initialize the chatbot with Gemini embedding model.
        """
        self.model_name = model_name
        self.index = None
        self.documents = []
        self.dimension = 768  # Dimension for text-embedding-004
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.max_num_result = 20  # Maximum number of results to return
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

    def load_vector_store(self, store_id: str):
        """
        Load vector store from disk.
        """
        try:
            index_path = f'vector_stores/{store_id}.index'
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Không tìm thấy file vector store: {index_path}")
            
            self.index = faiss.read_index(index_path)
            if self.index.d != self.dimension:
                raise ValueError(f"Kích thước vector ({self.index.d}) không khớp với kích thước mô hình ({self.dimension})")
            
            docs_path = f'vector_stores/{store_id}_documents.json'
            if not os.path.exists(docs_path):
                raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {docs_path}")
            
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
                
            logger.info(f"Đã tải thành công {len(self.documents)} sản phẩm điện thoại")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải vector store: {str(e)}")
            raise

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
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
            logger.error(f"Lỗi khi tạo vector nhúng: {str(e)}")
            return None

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search the vector store with a query.
        """
        if not self.documents:
            logger.warning("Chưa có dữ liệu điện thoại được tải")
            return []

        try:
            # Use max_num_result if top_k is not specified
            if top_k is None:
                top_k = self.max_num_result

            # Check for color search
            color_query = self._extract_color(query)
            if color_query:
                logger.info(f"Tìm kiếm điện thoại màu: {color_query}")
                results = []
                for doc in self.documents:
                    metadata = doc['metadata']
                    if 'color' in metadata and color_query.lower() in metadata['color'].lower():
                        results.append({
                            'document': doc,
                            'score': 1.0
                        })
                return results[:top_k]

            # If not a color search, use vector search
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                logger.warning("Không thể tạo vector nhúng cho câu truy vấn")
                return []

            # Vector Search
            distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    score = float(1 / (1 + distances[0][i]))
                    results.append({
                        'document': doc,
                        'score': score
                    })
            
            return results

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm: {str(e)}")
            return []

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate a response using Gemini model based on search results.
        """
        try:
            if not context:
                return "Xin lỗi, tôi không tìm thấy thông tin điện thoại phù hợp với yêu cầu của bạn."

            # Prepare context for Gemini
            context_text = f"Đây là thông tin của {len(context)} điện thoại liên quan:\n\n"
            for i, result in enumerate(context, 1):
                doc = result['document']
                metadata = doc['metadata']
                context_text += f"{i}. {metadata['title']}\n"
                context_text += f"   Giá: {metadata['price']:,} VND\n"
                context_text += f"   Thông số: {metadata['specs']}\n"
                context_text += f"   Khuyến mãi: {metadata['promotion']}\n"
                context_text += f"   Mô tả: {metadata['description']}\n\n"

            # Create prompt for Gemini
            prompt = f"""Bạn là một trợ lý mua sắm điện thoại thông minh. Hãy trả lời câu hỏi của người dùng dựa trên thông tin điện thoại được cung cấp.
            Nếu thông tin không có trong dữ liệu, hãy trả lời lịch sự rằng không tìm thấy thông tin.
            Hãy giữ câu trả lời ngắn gọn và tập trung vào thông tin quan trọng nhất.
            Nếu có nhiều kết quả, hãy tổ chức thông tin một cách rõ ràng và dễ đọc.
            Trả lời bằng tiếng Việt.

            Thông tin điện thoại:
            {context_text}

            Câu hỏi của người dùng: {query}

            Hãy trả lời một cách hữu ích và ngắn gọn:"""

            # Generate response
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            if not response.text:
                return "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại với cách diễn đạt khác."
                
            return response.text.strip()

        except Exception as e:
            logger.error(f"Lỗi khi tạo câu trả lời: {str(e)}")
            return "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn."

    def chat(self, query: str) -> str:
        """
        Main chat function that combines search and response generation.
        """
        try:
            if not query.strip():
                return "Vui lòng nhập câu hỏi về điện thoại."

            # Search for relevant information
            search_results = self.search(query)
            
            # Generate response based on search results
            response = self.generate_response(query, search_results)
            return response

        except Exception as e:
            logger.error(f"Lỗi trong quá trình chat: {str(e)}")
            return "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn."

def main():
    # Initialize chatbot
    chatbot = PhoneChatbot()
    store_id = "phone_vector_store_50_gemini"
    
    try:
        # Load vector store
        print("Đang tải dữ liệu điện thoại...")
        chatbot.load_vector_store(store_id)
        print("\n=== Trợ Lý Mua Sắm Điện Thoại ===")
        print("Tôi có thể giúp bạn tìm thông tin điện thoại, so sánh các mẫu và trả lời câu hỏi về thông số kỹ thuật.")
        print("Gõ 'quit', 'exit' hoặc 'bye' để kết thúc cuộc trò chuyện.")
        print("Gõ 'help' để xem hướng dẫn.")
        
        while True:
            try:
                # Get user input
                query = input("\nBạn: ").strip()
                
                # Check for quit command
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("\nCảm ơn bạn đã sử dụng Trợ Lý Mua Sắm Điện Thoại. Tạm biệt!")
                    break
                
                # Check for help command
                if query.lower() == 'help':
                    print("\nTôi có thể giúp bạn với:")
                    print("- Tìm điện thoại theo thông số (ví dụ: 'Cho tôi xem điện thoại màn hình 4.7 inch')")
                    print("- So sánh giá (ví dụ: 'Điện thoại nào dưới 5 triệu đồng?')")
                    print("- Tìm kiếm tính năng (ví dụ: 'Điện thoại nào có camera tốt nhất?')")
                    print("- Câu hỏi chung về điện thoại trong cơ sở dữ liệu")
                    continue
                
                # Generate and print response
                response = chatbot.chat(query)
                print(f"\nTrợ lý: {response}")
                
            except KeyboardInterrupt:
                print("\n\nTạm biệt!")
                break
            except Exception as e:
                logger.error(f"Lỗi trong vòng lặp chat: {str(e)}")
                print("\nĐã xảy ra lỗi. Vui lòng thử lại.")
                
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng: {str(e)}")
        print(f"\nLỗi: {str(e)}")
        print("Vui lòng kiểm tra cấu hình và thử lại.")

if __name__ == "__main__":
    main()