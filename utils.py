from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch


# Set device function is not finished or implemented have it hardcoded to use both gpus
def set_device(gpu: int = None) -> str:
    if torch.cuda.is_available() and gpu is not None:
        device = f"cuda:{gpu}"
    else:
        device = "cpu"
    return device


def load_embedding_model(
        model_name: str = "dunzhang/stella_en_1.5B_v5", device=set_device(0)
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


def load_docs():
    loaded_chat_store = SimpleChatStore.from_persist_path(
        persist_path="data/chatstorage.json"
    )
    docs = loaded_chat_store
    return docs


def load_environment_and_models():
    lc_embedding_model = load_embedding_model()
    embed_model = LangchainEmbedding(lc_embedding_model)
    llm = Ollama(model="llama3.1:latest", request_timeout=30.0, device=set_device(1))
    return embed_model, llm


def setup_index_and_query_engine(docs, embed_model, llm):
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
    return query_engine


def setup_index_and_chat_engine(docs, embed_model, llm):
    chat_store = SimpleChatStore()
    chat_store.persist(persist_path="data/chatstorage.json")
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900,
                                            chat_store=chat_store,
                                            chat_store_key="user1")
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    Settings.llm = llm
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.BEST,
        memory=memory,
        llm=llm,
        context_prompt=("Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "You are an Health Assistant that is designed to help patients on their fitness journey\n"
                        "Given the context information above I want you to think step by step to answer \n"
                        "the query in a clear and concise manner and incase case you don't know the answer \n"
                        "say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: ")
    )
    return chat_engine
