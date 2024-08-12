import HealthG as gr
import queue
import threading
from chat import main as healthg_main
from utils import handle_chat_storage


class HealthGradio:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.simple_chat_store, self.chat_memory = handle_chat_storage()
        self.user_id = "user1"
        self.message_index = 0

        # Start HealthG in a separate thread
        self.healthg_thread = threading.Thread(target=self.run_healthg)
        self.healthg_thread.start()

    def chat(self, message, history):
        print(f"Received message: {message}")  # Debug print
        self.input_queue.put(message)

        try:
            response = self.output_queue.get(timeout=60)
            print(f"Received response: {response}")  # Debug print

            # Save the message and response to chat history
            self.simple_chat_store.add_message(self.user_id, {"role": "user", "content": message}, self.message_index)
            self.message_index += 1
            self.simple_chat_store.add_message(self.user_id, {"role": "assistant", "content": response},
                                               self.message_index)
            self.message_index += 1

            # Persist the updated chat store
            self.simple_chat_store.persist("data/chat_storage.json")

            return response
        except queue.Empty:
            print("Timeout waiting for response")  # Debug print
            return "I'm sorry, I'm having trouble responding right now. Please try again."

    def run_healthg(self):
        def custom_input(*args):
            print("Waiting for input...")  # Debug print
            msg = self.input_queue.get()
            print(f"Received input: {msg}")  # Debug print
            return msg

        def custom_print(message):
            print(f"Sending output: {message}")  # Debug print
            self.output_queue.put(message)

        # Call the main function from HealthG with custom input and output functions
        healthg_main(custom_input=custom_input, custom_print=custom_print)

    def launch(self):
        # Load past chat history
        past_messages = self.simple_chat_store.get_messages(self.user_id)
        initial_chat = [(msg["content"], msg["content"]) for msg in past_messages if msg["role"] == "assistant"]

        iface = gr.ChatInterface(
            self.chat,
            chatbot=gr.Chatbot(value=initial_chat),
            textbox=gr.Textbox(label="HealthG", container=False, autoscroll=True,
                               placeholder="Type your health-related question here..."),
            title="HealthG: Your Personal Health Assistant",
            description="Welcome to HealthG! I'm here to assist you with health-related questions and advice. How can "
                        "I help you today?",
            show_progress="full",
            theme="soft",
            examples=[
                "What are some tips for maintaining a healthy diet?",
                "How can I improve my sleep quality?",
                "What are the benefits of regular exercise?",
                "How can I manage stress effectively?"
            ],
            fill_width=True,
            fill_height=True,
            retry_btn="Retry",
            undo_btn="Delete Last",
            clear_btn="Clear",
            stop_btn="Stop"
        )

        iface.launch(inbrowser=True, share=True)


if __name__ == "__main__":
    app = HealthGradio()
    app.launch()
