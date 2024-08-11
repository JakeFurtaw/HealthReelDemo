import torch
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Document
import os
from config import CHAT_STORAGE_PATH, TOKEN_LIMIT


def set_device(gpu: int = None) -> str:
    return f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"


def handle_chat_storage():
    if os.path.exists(CHAT_STORAGE_PATH) and os.path.getsize(CHAT_STORAGE_PATH) > 0:
        simple_chat_store = SimpleChatStore.from_persist_path(persist_path=CHAT_STORAGE_PATH)
    else:
        simple_chat_store = SimpleChatStore()
        simple_chat_store.persist(persist_path=CHAT_STORAGE_PATH)

    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT,
        chat_store=simple_chat_store
    )

    return simple_chat_store, chat_memory


def load_past_chats():
    if not os.path.exists(CHAT_STORAGE_PATH) or os.path.getsize(CHAT_STORAGE_PATH) == 0:
        print("Warning: Chat storage file is empty or doesn't exist.")
        return [Document(text="No chat history available.", metadata={"role": "system"})]

    loaded_chat_store = SimpleChatStore.from_persist_path(persist_path=CHAT_STORAGE_PATH)
    chats = []
    for user_id in loaded_chat_store.get_keys():
        messages = loaded_chat_store.get_messages(user_id)
        for idx, message in enumerate(messages):
            if isinstance(message, dict):
                chats.append(Document(text=message['content'], metadata={"role": message['role'], "index": idx}))
            elif isinstance(message, str):
                chats.append(Document(text=message, metadata={"role": "unknown", "index": idx}))
    return chats
