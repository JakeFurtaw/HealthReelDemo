from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage
from config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from utils import set_device


def load_embedding_model():
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME,
                                       device=set_device(0), trust_remote_code=True)
    return embed_model


def load_llm(model_name: str = LLM_MODEL_NAME):
    return Ollama(model=model_name, request_timeout=30.0, device=set_device(1), temperature=.75)


def setup_index_and_chat_engine(chats, embed_model, llm, memory):
    documents = []
    for chat in chats:
        if isinstance(chat, tuple) and len(chat) == 2:
            content, role = chat
            documents.append(Document(text=content, metadata={"role": role}))
        elif hasattr(chat, 'content') and hasattr(chat, 'role'):
            documents.append(Document(text=chat.content, metadata={"role": chat.role}))
        else:
            print(f"Skipping invalid chat entry: {chat}")

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    Settings.llm = llm

    chat_prompt = (
        "You are Health Bot, an advanced AI health assistant designed to help patients on their fitness \n"
        "and wellness journey. Your knowledge spans across fitness, nutrition, mental health, and general \n"
        "well-being. Always strive to provide accurate, helpful, and encouraging advice. If you're unsure \n"
        "about something, don't hesitate to say 'I'm not certain about that' and suggest consulting with a \n"
        "healthcare professional. Remember to be empathetic and supportive in your interactions. \n"
        "Your goal is to guide users towards healthier lifestyles while ensuring they seek professional \n"
        "medical advice when necessary. Only discuss health related topics, even if a user asks about another topic."
    )

    system_message = ChatMessage(role="system", content=chat_prompt)
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory,
        llm=llm,
        system_prompt=system_message,
        context_prompt=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and your role as HealthBot, please provide a helpful response \n"
            "to the user's query.\n"
            "Human: {query_str}\n"
            "Health Bot: "
        )
    )
    return chat_engine
