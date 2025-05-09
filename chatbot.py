import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define vector store ID
vector_store_id = "phone_vector_store_50_20240508_083000"  # ⚠️ Sửa lại cho đúng ID vector store của bạn

class ProductChatbot:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables.")
        self.client = OpenAI(api_key=self.api_key)
        self.assistant = None
        self.thread = None

    def create_assistant(self) -> None:
        try:
            self.assistant = self.client.beta.assistants.create(
                name="Product Assistant",
                instructions=(
                    "You are a helpful product assistant that answers questions about products. "
                    "Use the provided knowledge to find accurate information about products. "
                    "Always be clear, include prices in VND, and list features as bullet points when possible."
                ),
                model="gpt-4o",
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
            )
            logger.info("Assistant created successfully.")
        except Exception as e:
            logger.error(f"Failed to create assistant: {str(e)}")
            raise

    def create_thread(self, initial_message: Optional[str] = None) -> None:
        try:
            messages = [{"role": "user", "content": initial_message}] if initial_message else []
            self.thread = self.client.beta.threads.create(messages=messages)
            logger.info("Thread created.")
        except Exception as e:
            logger.error(f"Error creating thread: {str(e)}")
            raise

    def send_message(self, message: str) -> Dict[str, Any]:
        try:
            if not self.thread:
                self.create_thread()

            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=message
            )

            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id
            )

            messages = list(self.client.beta.threads.messages.list(thread_id=self.thread.id, run_id=run.id))
            if messages:
                return {
                    "success": True,
                    "response": messages[0].content[0].text.value,
                    "run_id": run.id
                }
            else:
                return {"success": False, "error": "No response received."}
        except Exception as e:
            logger.error(f"Error in send_message: {str(e)}")
            return {"success": False, "error": str(e)}

def print_welcome():
    print("\n" + "="*60)
    print("🤖 Welcome to the Product Assistant Chatbot!")
    print("="*60)
    print("Ask me about phones, prices, features, promotions, etc.")
    print("Type 'quit' to exit, 'help' for help, or 'clear' to clear screen.")
    print("-"*60)

def print_help():
    print("\nAvailable commands:")
    print("  help    - Show this help message")
    print("  quit    - End the conversation")
    print("  clear   - Clear the screen")
    print("\nExample questions:")
    print("  - What is the price of iPhone 13?")
    print("  - Show me phones under 10 million VND")
    print("  - What are the features of Samsung Galaxy A34?")
    print("-"*60)

def print_message(role: str, content: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] {'👤 You:' if role == 'user' else '🤖 Assistant:'}")
    print("-"*60)
    print(content.strip())
    print("-"*60)

def main():
    try:
        print("Initializing chatbot...")
        bot = ProductChatbot()
        bot.create_assistant()
        bot.create_thread()

        print_welcome()

        while True:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye! 👋")
                break
            elif user_input.lower() == "help":
                print_help()
                continue
            elif user_input.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                print_welcome()
                continue

            print("\n🤖 Assistant is typing", end="", flush=True)
            for _ in range(3):
                time.sleep(0.5)
                print(".", end="", flush=True)
            print()

            response = bot.send_message(user_input)
            if response["success"]:
                print_message("assistant", response["response"])
            else:
                print_message("assistant", f"Error: {response['error']}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
