from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key từ .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Đường dẫn đến các file FAISS và metadata JSON đã lưu
INDEX_FILE_PATH = "vector_stores/phone_vector_store_50_1735471621602502345.index"
DOCS_FILE_PATH = "vector_stores/phone_vector_store_50_1735471621602502345_documents.json"

def upload_vector_store():
    try:
        print("🔄 Uploading index file...")
        # Upload file với mục đích 'user_data'
        index_file = client.files.create(file=open(INDEX_FILE_PATH, "rb"), purpose="user_data")
        print(f"Index file uploaded with ID: {index_file.id}")
        
        print("🔄 Uploading document metadata...")
        doc_file = client.files.create(file=open(DOCS_FILE_PATH, "rb"), purpose="user_data")
        print(f"Document file uploaded with ID: {doc_file.id}")

        print("📦 Creating vector store...")
        vector_store = client.beta.vector_stores.create(
            name="Phone Vector Store",
            files=[index_file.id, doc_file.id]
        )

        print("✅ Vector store created successfully!")
        print(f"🆔 Vector Store ID: {vector_store.id}")
    except Exception as e:
        print(f"❌ Error uploading vector store: {e}")

if __name__ == "__main__":
    upload_vector_store()
