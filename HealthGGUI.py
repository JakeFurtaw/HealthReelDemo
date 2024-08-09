import json
import gradio as gr
import queue
import threading
from HealthG import main as healthg_main
from utils import handle_chat_storage


class HealthGradio:
    def __init__(self):
        self.message_index = 0
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.simple_chat_store, self.chat_memory = handle_chat_storage()
        self.user_id = "user1"
        self.chatbot = []

        # Start HealthG in a separate thread
        self.healthg_thread = threading.Thread(target=self.run_healthg)
        self.healthg_thread.start()

        try:
            with open("data/chat_storage.json", "r") as f:
                chat_history = json.load(f)
                if self.user_id in chat_history:
                    messages = chat_history[self.user_id]
                    # Display past conversation history in chat window
                    for message in messages:
                        self.chatbot.append((message["content"], "" if message["role"] == "user" else message["content"]))
        except FileNotFoundError:
            pass

    def chat(self, message, history):
        self.input_queue.put(message)
        try:
            response = self.output_queue.get(timeout=60)
            # Update history
            history.append((message, response))
            return "", history  # Return empty string for input and updated history
        except queue.Empty:
            return "", history + [(message, "I'm sorry, I'm having trouble responding right now. Please try again.")]

    def run_healthg(self):
        def custom_input(*args):
            msg = self.input_queue.get()
            return msg

        def custom_print(message):
            self.output_queue.put(message)

        # Call the main function from HealthG with custom input and output functions
        healthg_main(custom_input=custom_input, custom_print=custom_print)

    def start_new_chat(self):
        # Clear the messages for the user, reset index, and reset chat memory
        self.simple_chat_store.delete_messages(self.user_id)
        self.message_index = 0
        self.chat_memory.reset()
        # Persist the empty chat store
        self.simple_chat_store.persist("data/chat_storage.json")
        # Empty Queue Just Incase
        self.input_queue.empty()
        self.output_queue.empty()

        return []

    def launch(self):
        with gr.Blocks(theme="soft", fill_height=True) as iface:
            gr.Markdown("# HealthG: Your Personal Health Assistant")
            gr.Markdown(
                "Welcome to HealthG! I'm here to assist you with health-related questions and advice. How can I help "
                "you today?")
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="HealthG", container=False, autoscroll=True, autofocus=True,
                             placeholder="Type your health-related question here...")
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot])  # Just clears chat window
                new_chat = gr.Button("Start New Chat")  # Clears chat history too

            msg.submit(self.chat, [msg, chatbot], [msg, chatbot])
            new_chat.click(self.start_new_chat, outputs=[chatbot])

            gr.Examples(
                examples=[
                    "What are some tips for maintaining a healthy diet?",
                    "How can I improve my sleep quality?",
                    "What are the benefits of regular exercise?",
                    "How can I manage stress effectively?"
                ],
                inputs=msg
            )

        iface.launch(inbrowser=True, share=True)


if __name__ == "__main__":
    app = HealthGradio()
    app.launch()
