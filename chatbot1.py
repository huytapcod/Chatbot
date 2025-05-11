import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from search_vector_store import VectorStoreSearcher
from build_vector_store import VectorStoreBuilder
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    raise ValueError("GEMINI_API_KEY not found in environmet variables.")

# Configure Gemini API
try:
    genai.configure(api_key=api_key)
    logger.info("Successfully configured Gemini API")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise

class ChatbotWithVectorStore:
    def __init__(self, vectorstore_id: str, model_name: str = "text-embedding-004"):
        """Initialize chatbot with vector store."""
        logger.debug(f"Initializing chatbot with vectorstore_id: {vectorstore_id}")
        self.vectorstore_id = vectorstore_id
        self.searcher = VectorStoreSearcher(model_name=model_name)
        self.conversation_history = []
        try:
            self.searcher.load_vector_store(vectorstore_id)
            logger.info(f"Successfully loaded vector store: {vectorstore_id}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Use Gemini to analyze query and extract filters."""
        logger.debug(f"Analyzing query: {query}")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Include conversation history in the prompt
            history_context = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" 
                                       for msg in self.conversation_history[-3:]])  # Last 3 exchanges
            prompt = f"""
            Bạn là trợ lý phân tích câu hỏi. Phân tích câu hỏi sau và trích xuất các bộ lọc (filters) như loại sản phẩm, màu sắc, giá tối thiểu, giá tối đa, khuyến mãi, hoặc thương hiệu. Trả về dưới dạng JSON. Nếu không tìm thấy bộ lọc, trả về trường rỗng. Đảm bảo phản hồi là JSON hợp lệ.

            Lịch sử hội thoại gần đây:
            {history_context}

            Câu hỏi hiện tại: {query}

            Ví dụ:
            - "Điện thoại màu đen dưới 10 triệu" → {{"type": "điện thoại", "color": "đen", "max_price": 10000000}}
            - "Giảm 2 triệu" → {{"promotion": "giảm 2 triệu"}}

            Trả về:
            """
            response = model.generate_content(prompt)
            try:
                filters = json.loads(response.text.strip('```json\n').strip('```'))
                logger.debug(f"Extracted filters: {filters}")
                return filters
            except json.JSONDecodeError:
                logger.warning("Failed to parse Gemini response as JSON. Returning empty filters.")
                return {}
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {}

    def get_answer_from_vector_store(self, query: str, top_k: int = 10):
        """Get answer from vector store using query and extracted filters."""
        logger.debug(f"Searching vector store for query: {query}")
        try:
            # Analyze query to extract filters
            filters = self.analyze_query(query)
            results = self.searcher.search(query, filters=filters, top_k=top_k)
            
            if not results:
                logger.info("No results found in vector store. Falling back to Gemini.")
                return self.get_answer_from_gemini(query)
            
            # Format the response
            response = "Tôi tìm thấy các sản phẩm phù hợp với yêu cầu của bạn:\n"
            for i, result in enumerate(results, 1):
                doc = result['document']
                score = result['score']
                metadata = doc['metadata']
                
                # Format price with commas
                price = f"{metadata['price']:,}".replace(',', '.')
                
                response += (f"\n{i}. {metadata['title']} (Độ phù hợp: {score:.2f})\n"
                           f"   - Giá: {price} VND\n"
                           f"   - Màu: {metadata['color']}\n"
                           f"   - Khuyến mãi: {metadata['promotion']}\n"
                           f"   - Thông số: {metadata['specs']}\n"
                           f"   - Mô tả: {metadata['description']}\n")
            
            # Add follow-up suggestion
            response += "\nBạn có muốn biết thêm thông tin về sản phẩm nào không?"
            
            logger.debug(f"Returning {len(results)} results from vector store")
            return response
        
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return self.get_answer_from_gemini(query)

    def get_answer_from_gemini(self, query: str):
        """Fallback to Gemini for answering."""
        logger.debug("Querying Gemini API")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Include conversation history in the prompt
            history_context = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" 
                                       for msg in self.conversation_history[-3:]])  # Last 3 exchanges
            
            prompt = f"""
            Bạn là trợ lý bán hàng thông minh. Dựa vào lịch sử hội thoại và câu hỏi hiện tại, hãy trả lời tự nhiên và hữu ích bằng tiếng Việt.

            Lịch sử hội thoại gần đây:
            {history_context}

            Câu hỏi hiện tại: {query}

            Hãy trả lời:
            """
            
            response = model.generate_content(prompt)
            logger.info("Successfully queried Gemini API")
            return f"Chatbot (Gemini): {response.text.strip()}"
        except Exception as e:
            logger.error(f"Error querying Gemini: {str(e)}")
            return "Gemini API không khả dụng. Vui lòng kiểm tra API key hoặc quota."

    def process_message(self, user_input: str) -> str:
        """Process user message and maintain conversation history."""
        response = self.get_answer_from_vector_store(user_input)
        # Store the conversation
        self.conversation_history.append({
            'user': user_input,
            'assistant': response
        })
        return response

def build_vector_store(vectorstore_id: str, data_path: str = "data/phonedatas (3).csv"):
    """Build a new vector store if it doesn't exist."""
    logger.debug(f"Building vector store with ID: {vectorstore_id}")
    try:
        builder = VectorStoreBuilder(model_name="text-embedding-004")
        documents = builder.load_phone_data(data_path)
        new_store_id = builder.build_vector_store(documents)
        logger.info(f"Created new vector store with ID: {new_store_id}")
        return new_store_id
    except Exception as e:
        logger.error(f"Error building vector store: {str(e)}")
        raise

def chat():
    """Chat with the user via terminal."""
    logger.debug("Starting chat function")
    print("Chào mừng bạn đến với Chatbot AI (Dùng dữ liệu VectorStore)! (Nhập 'exit' để thoát)")
    
    vectorstore_id = "phone_vector_store_50_gemini"
    data_path = "data/phonedatas (3).csv"

    if not (os.path.exists(f"vector_stores/{vectorstore_id}.index") and 
            os.path.exists(f"vector_stores/{vectorstore_id}_documents.json")):
        print(f"Vector store '{vectorstore_id}' không tồn tại. Đang tạo mới...")
        try:
            vectorstore_id = build_vector_store(vectorstore_id, data_path)
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            print(f"Lỗi khi tạo vector store: {str(e)}")
            return

    try:
        chatbot = ChatbotWithVectorStore(vectorstore_id, model_name="text-embedding-004")
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        print(f"Lỗi khi khởi tạo chatbot: {str(e)}")
        return

    while True:
        user_input = input("Bạn: ")
        logger.debug(f"User input: {user_input}")
        if user_input.lower() == "exit":
            print("Tạm biệt!")
            logger.info("Exiting chat")
            break
        
        response = chatbot.process_message(user_input)
        print(f"Chatbot: {response}\n")
        logger.debug(f"Chatbot response: {response}")

if __name__ == "__main__":
    try:
        chat()
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"Unexpected error: {str(e)}")