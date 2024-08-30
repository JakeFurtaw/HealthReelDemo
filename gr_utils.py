from datetime import datetime
from config import CHAT_STORAGE_PATH, LONG_TERM_STORAGE_PATH
from models import load_models, setup_index_and_chat_engine
from utils import handle_chat_storage
from llama_index.core.llms import MessageRole, ChatMessage


class HealthBotGradio:
    def __init__(self):
        self.user_id = None
        self.embed_model, self.llm = load_models()
        self.simple_chat_store, self.chat_memory = handle_chat_storage()
        self.chat_engine = None

    def set_user_id(self, user_id):
        self.user_id = user_id
        self.chat_engine= setup_index_and_chat_engine(
            self.simple_chat_store.get_messages(self.user_id),
            self.embed_model,
            self.llm,
            self.chat_memory,
        )
        return self.user_id, self.get_past_messages()

    def get_past_messages(self):
        messages = self.simple_chat_store.get_messages(key=self.user_id)
        return [(msg.content, next_msg.content) for msg, next_msg in zip(messages[::2], messages[1::2] + [None])]

    def chat(self, message, history):
        user_message = ChatMessage(role=MessageRole.USER, content=message)
        self.simple_chat_store.add_message(key=self.user_id, message=user_message)
        history.append((message, ""))

        full_response = ""
        for delta in self.chat_engine.stream_chat(message).response_gen:
            full_response += delta
            history[-1] = (message, full_response)
            yield "", history

        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=full_response)
        self.simple_chat_store.add_message(key=self.user_id, message=assistant_message)

        self.simple_chat_store.persist(persist_path=CHAT_STORAGE_PATH)
        self.simple_chat_store.persist(persist_path=LONG_TERM_STORAGE_PATH)

    def start_new_chat(self):
        if self.chat_engine:
            self.simple_chat_store.delete_messages(self.user_id)
            self.simple_chat_store.persist(persist_path=CHAT_STORAGE_PATH)
            self.chat_memory.reset()
            self.chat_engine = setup_index_and_chat_engine(
                self.simple_chat_store.get_messages(self.user_id),
                self.embed_model,
                self.llm,
                self.chat_memory,
            )
        return []