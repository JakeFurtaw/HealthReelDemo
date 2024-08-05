from utils import (
    setup_index_and_chat_engine, load_environment_and_models, load_past_chats, handle_chat_storage
)
from llama_index.core.llms import MessageRole, ChatMessage


def main(custom_input=input, custom_print=print):
    user_id = "user1"
    embed_model, llm = load_environment_and_models()
    chats = load_past_chats()
    simple_chat_store, chat_memory = handle_chat_storage()
    message_index = 0

    chat_engine = setup_index_and_chat_engine(chats, embed_model, llm, chat_memory)
    past_messages = simple_chat_store.get_messages(key=user_id)
    for msg in past_messages:
        chat_memory.put(ChatMessage(role=MessageRole(msg['role']), content=msg['content']))

    custom_print("Welcome to HealthG, your personal health assistant.")

    while True:
        user_query = custom_input()
        print(f"Received query: {user_query}")  # Debug print

        if user_query.lower() == 'e':
            simple_chat_store.delete_last_message(key=user_id)
            chat_memory.reset()
            custom_print("Thanks for using HealthG. Goodbye!")
            break

        new_message = ChatMessage(role=MessageRole.USER, content=user_query)
        simple_chat_store.add_message(key=user_id, message=new_message, idx=message_index)
        chat_memory.put(new_message)
        message_index += 1

        response = chat_engine.chat(user_query)
        print(f"Generated response: {response}")  # Debug print
        custom_print(str(response))

        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=str(response))
        simple_chat_store.add_message(key=user_id, message=assistant_message, idx=message_index)
        chat_memory.put(assistant_message)
        message_index += 1


if __name__ == "__main__":
    main()
