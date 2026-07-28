import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  
MODEL_NAME = "google/gemma-4-26b-a4b-it:free"
APP_TITLE = "Nyahu AI"
