from llama_index.core.chat_engine.types import ChatMode
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch


# Set device function is not finished or implemented have it hardcoded to use both gpus
def set_device():
    if torch.cuda_is_available():
        device = "cuda"
    else:
        device = "cpu"


def set_llm():
    llm = Ollama(model="mistral-nemo:latest",
                 request_timeout=30.0,
                 device="cuda:0")
    return llm


def set_embed_model():
    embed_model = HuggingFaceEmbedding(
        model_name="dunzhang/stella_en_400M_v5",
        device="cuda:1",
        trust_remote_code=True)
    return embed_model


# Saving user data to a json file
def load_docs():
    documents = SimpleChatStore.from_persist_path(persist_path="/data/chat_store.json")
    return documents


def setup_index_and_chat_engine(documents, embed_model, llm):
    # Memory by user, change chat store key to
    chat_store = SimpleChatStore()
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900,
                                            chat_store=chat_store,
                                            chat_store_key="user1")  # Swap Users here
    chat_store.persist(persist_path="/data/chat_store.json")  # Saves chat logs to json file
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
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
