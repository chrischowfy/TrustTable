# configs/config.py
import os

class Config:
    # Read credentials from the environment. Do not commit API keys.
    API_KEY = os.getenv("LLM_API_KEY", "")
    BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    MODEL_NAME = os.getenv("LLM_MODEL", "deepseek-chat")
    TEMPERATURE = 0.0
    
    # Z3 timeout in milliseconds.
    Z3_TIMEOUT = 5000 
