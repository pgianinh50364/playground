from fastapi import WebSocket
from typing import Dict, Optional

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.api_key: Dict[str, str] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"Client {client_id} connected!")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.api_key:
            del self.api_key[client_id]

    def set_api_key(self, client_id: str, api_key: str):
        self.api_key[client_id] = api_key
    
    def get_api_key(self, client_id: str) -> Optional[str]:
        return self.api_key.get(client_id)

    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)