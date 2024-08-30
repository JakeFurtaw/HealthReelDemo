from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage
from config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from utils import set_device
from llama_index.core import (VectorStoreIndex, get_response_synthesizer, Document)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

def load_models():
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME,
                                       device=set_device(0), trust_remote_code=True)
    llm = Ollama(model=LLM_MODEL_NAME, request_timeout=30.0, device=set_device(1), temperature=.75)
    return embed_model, llm


def initialize_chroma_database(long_term_storage_path, collection_name="health_bot_conversations"):
    chroma_client = chromadb.PersistentClient(path=long_term_storage_path)
    try:
        collection = chroma_client.get_collection(collection_name)
    except ValueError:
        collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store


def setup_index_and_chat_engine(chats, embed_model, llm, memory, vector_store):
    # Create documents from current chat
    current_documents = [Document(text=chat.content, metadata={"role": chat.role})
                         for chat in chats if hasattr(chat, 'content') and hasattr(chat, 'role')]
    # Create the index with the current documents and long-term storage
    index = VectorStoreIndex.from_documents(current_documents, vector_store=vector_store, embed_model=embed_model)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3, embed_model=embed_model)
    response_synthesizer = get_response_synthesizer(streaming=True, llm=llm, response_mode=ResponseMode.COMPACT)
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
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
        query_engine=query_engine,
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

def save_to_long_term_storage(vector_store, document):
    vector_store.add_documents([document])
