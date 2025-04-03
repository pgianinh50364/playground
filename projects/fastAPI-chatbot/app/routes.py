import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
from .connection_manager import ConnectionManager
from .chatbot import Chatbot

logger = logging.getLogger(__name__)
router = APIRouter()

connection_manager = ConnectionManager()
chatbot = Chatbot()

templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await connection_manager.connect(websocket, client_id)
    try:
        await connection_manager.send_message(
            "Welcome to the AI chatbot! Please set your OpenAI API key to get started.",
            client_id
        )
        while True:

            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "message")
                content = message_data.get("content", "")

                if message_type == "api_key":
                    api_key = content
                    connection_manager.set_api_key(client_id, api_key)
                    success = chatbot.set_api_key(client_id, api_key)
                    if success:
                        await connection_manager.send_message(client_id, "API key set successfully")
                    else:
                        await connection_manager.send_message(client_id, "Invalid API key.")

                elif message_type == "message":
                    api_key = connection_manager.get_api_key(client_id)
                    if api_key:
                        response = await chatbot.get_response(client_id, content)
                    else:
                        response = chatbot.fallback_response(content)
                    
                    await connection_manager.send_message(client_id, response)

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
                await connection_manager.send_message("Invalid message format", client_id)
    
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")