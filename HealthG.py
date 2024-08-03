from utils import (
    setup_index_and_chat_engine, load_environment_and_models
)


def main() -> None:
    embed_model, llm = load_environment_and_models()
    # Chat Engine, Added Memory
    chat_engine = setup_index_and_chat_engine(docs, embed_model, llm)
    print("Welcome to HealthG, your personal health assistant.")
    while True:
        user_query = input("Ask Me Anything: ")
        if user_query.lower() == 'e':
            print("Thanks for using HealthG. Goodbye!")
            break
        response = chat_engine.chat(user_query)
        print("Response:", response)
        print("\n" + "-" * 100 + "\n")


if __name__ == "__main__":
    main()
