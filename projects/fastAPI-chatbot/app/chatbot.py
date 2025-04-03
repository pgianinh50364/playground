import logging
from typing import Dict
from .client import InferClient

logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self):
        self.clients: Dict[str, InferClient] = {}

    def get_client(self, session_id: str) -> InferClient:
        if session_id not in self.clients:
            self.clients[session_id] = InferClient()
        return self.clients[session_id]
    
    def set_api_key(self, session_id: str, api_key: str) -> bool:
        client = self.get_client(session_id)
        return client.setup_client(api_key)
    
    async def get_response(self, session_id: str, message: str) -> str:
        client = self.get_client(session_id)

        if not client.api_key:
            return "API key not set. Please set the API key first."
        
        try:
            response = client.get_ai_response(message)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return f"Encountered an error: {str(e)}"
        
    def fallback_response(self, message: str) -> str:
        message = message.lower()

        if "hello" in message or "hi" in message:
            return "Hello! I can assist if only you provide me with an API key"        
        elif "help" in message:
            return "I'm an AI chatbot. Please set an API key."
        else:
            return "I need an HF API key to assist you. Please set it first."