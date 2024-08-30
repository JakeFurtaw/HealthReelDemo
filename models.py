from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage
from config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from utils import set_device
from llama_index.core import (VectorStoreIndex, Document)


def load_models():
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME,
                                       device=set_device(0), trust_remote_code=True)
    llm = Ollama(model=LLM_MODEL_NAME, request_timeout=30.0, device=set_device(1), temperature=.75)
    return embed_model, llm

def setup_index_and_chat_engine(chats, embed_model, llm, memory):
    # Create documents from current chat
    current_chats = [Document(text=chat.content, metadata={"role": chat.role})
                         for chat in chats if hasattr(chat, 'content') and hasattr(chat, 'role')]
    index = VectorStoreIndex.from_documents(current_chats, embed_model=embed_model)
    chat_prompt = (
        "You are Health Bot, an advanced AI health assistant designed to help patients on their fitness "
        "and wellness journey. Your knowledge spans across fitness, nutrition, mental health, and general "
        "well-being. Always strive to provide accurate, helpful, and encouraging advice. If you're unsure "
        "about something, don't hesitate to say 'I'm not certain about that' and suggest consulting with a "
        "healthcare professional. Remember to be empathetic and supportive in your interactions. "
        "Your goal is to guide users towards healthier lifestyles while ensuring they seek professional "
        "medical advice when necessary. Only discuss health related topics, even if a user asks about another topic."
    )
    system_message = ChatMessage(role="system", content=chat_prompt)
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory,
        system_prompt=system_message,
        llm=llm,
        context_prompt=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and your role as HealthBot, please provide a helpful response "
            "to the user's query. If the context includes relevant information from past sessions, incorporate it wisely.\n"
            "Human: {query_str}\n"
            "Health Bot: "
        )
    )
    return chat_engine
