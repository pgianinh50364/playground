import logging
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import router
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="FastAPI LLama Chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)

if __name__ == "__main__":
    # Add more detailed logging
    logger.info(f"Starting server on {config.HOST}:{config.PORT}")
    logger.info(f"WebSocket endpoint available at ws://{config.HOST}:{config.PORT}/ws/chat")
    uvicorn.run(
        "main:app", 
        host=config.HOST, 
        port=config.PORT, 
        reload=config.DEBUG,
    )