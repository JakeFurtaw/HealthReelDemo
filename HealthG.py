from llama_index.core.memory import ChatMemoryBuffer
from utils import load_past_chats, handle_chat_storage
from models import load_embedding_model, load_llm, setup_index_and_chat_engine
from llama_index.core.llms import MessageRole, ChatMessage
from config import CHAT_STORAGE_PATH


class HealthG:
    def __init__(self, user_id):
        self.user_id = user_id
        self.embed_model = load_embedding_model()
        self.llm = load_llm()
        self.chats = load_past_chats()
        self.simple_chat_store, self.chat_memory = handle_chat_storage()
        self.message_index = len(self.simple_chat_store.get_messages(self.user_id))
        self.chat_memory = self._load_past_messages()
        self.chat_engine = setup_index_and_chat_engine(self.chats, self.embed_model, self.llm, self.chat_memory)

    def _load_past_messages(self):
        past_messages = self.simple_chat_store.get_messages(key=self.user_id)
        updated_chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.chat_memory.token_limit,
            chat_store=self.simple_chat_store
        )
        for msg in past_messages:
            if isinstance(msg, ChatMessage):
                updated_chat_memory.put(msg)
            elif isinstance(msg, dict):
                updated_chat_memory.put(ChatMessage(role=MessageRole(msg['role']), content=msg['content']))
            else:
                print(f"Unexpected message format: {msg}")

        # Update the message_index
        self.message_index = len(past_messages)

        return updated_chat_memory

    def get_past_messages(self):
        past_messages = self.simple_chat_store.get_messages(key=self.user_id)
        formatted_messages = []
        user_message = None

        for msg in past_messages:
            if msg.role == MessageRole.USER:
                user_message = msg.content
            elif msg.role == MessageRole.ASSISTANT and user_message is not None:
                formatted_messages.append((user_message, msg.content))
                user_message = None

        # If there's an unpaired user message at the end, add it
        if user_message is not None:
            formatted_messages.append((user_message, None))

        return formatted_messages

    def chat(self, user_query):
        self.message_index += 1
        user_message = ChatMessage(role=MessageRole.USER, content=user_query)
        self.simple_chat_store.add_message(key=self.user_id, message=user_message, idx=self.message_index)
        response = self.chat_engine.chat(user_query)
        self.message_index += 1
        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=str(response))
        self.simple_chat_store.add_message(key=self.user_id, message=assistant_message, idx=self.message_index)
        self.simple_chat_store.persist(CHAT_STORAGE_PATH)
        self.chat_memory.put(user_message)
        self.chat_memory.put(assistant_message)
        return str(response)

    def reset_chat(self):
        self.simple_chat_store.delete_messages(self.user_id)
        self.message_index = 0
        self.chat_memory.reset()
        self.simple_chat_store.persist(CHAT_STORAGE_PATH)


def main(custom_input=input, custom_print=print):
    health_g = HealthG()
    while True:
        user_query = custom_input()
        if user_query.lower() == 'e':
            health_g.reset_chat()
            custom_print("Thanks for using HealthG. Goodbye!")
            break
        response = health_g.chat(user_query)
        custom_print(response)


if __name__ == "__main__":
    main()
