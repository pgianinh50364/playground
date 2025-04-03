import logging
from huggingface_hub import InferenceClient
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class InferClient:
    def __init__(self, api_key: Optional[str]=None):
        self.api_key = api_key
        self.client = None
        self.chat_history = []

        if api_key:
            self.setup_client(api_key)
    
    def setup_client(self, api_key: str) -> bool:
        try:
            self.api_key = api_key
            self.client = InferenceClient(
                provider="hf-inference",
                api_key=api_key
            )
            logger.info("HF initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize HF client: {e}")
            return False
        
    def add_message_to_hist(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    def get_ai_response(self, message: str, model: str="mistralai/Mistral-7B-Instruct-v0.2", temperature: float=0.7, max_tokens: int=300) -> Dict[str, Any]:
        if not self.client:
            return "HF client is not initialized."
            
        try:
            self.add_message_to_hist("user", message)
            messages = [{"role": "system", "content": "You are a helpful assistant."}] + self.chat_history
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=500
            )

            ai_response = response.choices[0].message
            self.add_message_to_hist("assistant", ai_response)
            return response
        
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"