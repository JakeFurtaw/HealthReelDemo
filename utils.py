from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch
import os


def set_device(gpu: int = None) -> str:
    if torch.cuda.is_available() and gpu is not None:
        device = f"cuda:{gpu}"
    else:
        device = "cpu"
    return device


def load_embedding_model(
        model_name: str = "dunzhang/stella_en_400M_v5", device=set_device(0)
) -> HuggingFaceBgeEmbeddings:
    model_kwargs = {"device": device,
                    "trust_remote_code": True}
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model


def handle_chat_storage():
    # Create and persist SimpleChatStore
    simple_chat_store = SimpleChatStore()
    # -----------------------------------------------
    # -------Change storage location below-----------
    # -----------------------------------------------
    simple_chat_store.persist(persist_path="data/chat_storage.json")

    # Create ChatMemoryBuffer
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=6000,
        chat_store=simple_chat_store
    )

    return simple_chat_store, chat_memory


def load_past_chats():
    # -----------------------------------------------
    # -------Change storage location below-----------
    # -----------------------------------------------
    chat_store_path = "data/chat_storage.json"
    if not os.path.exists(chat_store_path) or os.path.getsize(chat_store_path) == 0:
        print("Warning: Chat storage file is empty or doesn't exist.")
        return [Document(text="No chat history available.", metadata={"role": "system"})]

    loaded_chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_path)
    chats = []
    for user_id in loaded_chat_store.get_keys():
        messages = loaded_chat_store.get_messages(user_id)
        for idx, message in enumerate(messages):
            if isinstance(message, dict):
                chats.append(Document(text=message['content'], metadata={"role": message['role'], "index": idx}))
            elif isinstance(message, str):
                chats.append(Document(text=message, metadata={"role": "unknown", "index": idx}))
    return chats


def load_environment_and_models():
    lc_embedding_model = load_embedding_model()
    embed_model = LangchainEmbedding(lc_embedding_model)
    llm = Ollama(model="mistral-nemo:latest", request_timeout=30.0, device=set_device(1))
    return embed_model, llm


def setup_index_and_query_engine(docs, embed_model, llm):
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
    return query_engine


def setup_index_and_chat_engine(chats, embed_model, llm, memory):
    index = VectorStoreIndex.from_documents(chats, embed_model=embed_model)
    Settings.llm = llm
    # Define the chat prompt
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
        max_tokens=100,
        num_output=100,
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
