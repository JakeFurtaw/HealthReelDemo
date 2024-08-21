import os

# Paths
DATA_DIR = "data"
CHAT_STORAGE_PATH = os.path.join(DATA_DIR, "chat_storage.json")

# Model settings
EMBEDDING_MODEL_NAME = "/home/jake/Programming/Models/embedding/stella_en_400M_v5"
LLM_MODEL_NAME = "mistral-nemo:latest"

# Chat settings
TOKEN_LIMIT = 115000

# Gradio settings
GRADIO_THEME = "soft"
CHATBOT_HEIGHT = 1000
