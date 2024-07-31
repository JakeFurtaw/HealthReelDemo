from utils import (
    set_llm, setup_index_and_chat_engine, set_embed_model, load_docs
)


def main() -> None:
    embed_model = set_embed_model()
    llm = set_llm()
    docs = load_docs()

    # Chat Engine, Added Memory
    chat_engine = setup_index_and_chat_engine(docs, embed_model, llm)

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
