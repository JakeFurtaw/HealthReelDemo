from utils import (
    setup_index_and_chat_engine, load_environment_and_models, load_past_chats, handle_chat_storage
)
from llama_index.core.chat_engine.types import ChatMessage


def main() -> None:
    # --------------------------------------
    # -------Change user ID below-----------
    # --------------------------------------
    user_id = "user1"
    embed_model, llm = load_environment_and_models()
    chats = load_past_chats()
    simple_chat_store, chat_memory = handle_chat_storage()

    # Chat Engine, Added Memory
    chat_engine = setup_index_and_chat_engine(chats, embed_model, llm, chat_memory)
    # Loading past user messages
    past_messages = simple_chat_store.get_messages(key=user_id)
    for msg in past_messages:
        chat_memory.put(ChatMessage(role=msg['role'], content=msg['content']))

    print("Welcome to HealthG, your personal health assistant.")
    while True:
        user_query = input("Ask Me Anything: ")
        # Adding new query to memory and storage
        new_message = ChatMessage(role="human", content=user_query)
        simple_chat_store.add_message(key=user_id, message=new_message)
        chat_memory.put(new_message)

        if user_query.lower() == 'e':
            # Removing chat if it's the exit key
            simple_chat_store.delete_last_message(key=user_id)
            # resetting chat memory at the end of the session
            chat_memory.reset()
            print("Thanks for using HealthG. Goodbye!")
            break

        response = chat_engine.chat(user_query)
        print("Response:", response)
        # Add assistant's response to memory and storage
        assistant_message = ChatMessage(role="assistant", content=str(response))
        simple_chat_store.add_message(key=user_id, message=assistant_message)
        chat_memory.put(assistant_message)
        print("\n" + "-" * 100 + "\n")


if __name__ == "__main__":
    main()
