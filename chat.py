from utils import handle_chat_storage
from models import setup_index_and_chat_engine
from config import CHAT_STORAGE_PATH


class Chat:
    def __init__(self, user_id, embed_model, llm):
        self.user_id = user_id
        self.embed_model = embed_model
        self.llm = llm
        self.simple_chat_store, self.chat_memory = handle_chat_storage()
        self.chat_engine = setup_index_and_chat_engine(self.simple_chat_store.get_messages(self.user_id),
                                                       self.embed_model, self.llm, self.chat_memory)

    def create_chat_engine(self):
        chats, memory = handle_chat_storage()
        embed_model = self.embed_model
        llm = self.llm
        return setup_index_and_chat_engine(chats=chats, llm=llm, embed_model=embed_model, memory=memory)

    def reset_chat(self):
        self.simple_chat_store.delete_messages(self.user_id)
        self.chat_memory.reset()
        self.simple_chat_store.persist(CHAT_STORAGE_PATH)
        self.chat_engine = self.create_chat_engine()
