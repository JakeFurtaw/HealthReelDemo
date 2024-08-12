from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage
from config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from utils import set_device


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME, device=set_device(0)):
    model_kwargs = {"device": device, "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return LangchainEmbedding(embedding_model)


def load_llm(model_name: str = LLM_MODEL_NAME):
    return Ollama(model=model_name, request_timeout=30.0, device=set_device(1))


def setup_index_and_chat_engine(chats, embed_model, llm, memory):
    documents = [
        Document(text=chat.content, metadata={"role": chat.role})
        for chat in chats
    ]

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    Settings.llm = llm

    chat_prompt = (
        "You are HealthG, an advanced AI health assistant designed to help patients on their fitness \n"
        "and wellness journey. Your knowledge spans across fitness, nutrition, mental health, and general \n"
        "well-being. Always strive to provide accurate, helpful, and encouraging advice. If you're unsure \n"
        "about something, don't hesitate to say 'I'm not certain about that' and suggest consulting with a \n"
        "healthcare professional. Remember to be empathetic and supportive in your interactions. \n"
        "Your goal is to guide users towards healthier lifestyles while ensuring they seek professional \n"
        "medical advice when necessary."
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
            "Given the context information and your role as HealthG, please provide a helpful response \n"
            "to the user's query.\n"
            "Human: {query_str}\n"
            "HealthG: "
        )
    )
    return chat_engine
