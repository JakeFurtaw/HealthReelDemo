import gradio
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
        self.past_messages = self.load_past_messages()

        # Start HealthG in a separate thread
        self.healthg_thread = threading.Thread(target=self.run_healthg)
        self.healthg_thread.start()

    def load_past_messages(self):
        messages = self.simple_chat_store.get_messages(self.user_id)
        chat_history = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    chat_history.append((msg["content"], None))
                elif msg["role"] == "assistant":
                    if chat_history and chat_history[-1][1] is None:
                        chat_history[-1] = (chat_history[-1][0], msg["content"])
                    else:
                        chat_history.append((None, msg["content"]))
        return chat_history

    def chat(self, message, history):
        self.input_queue.put(message)

        try:
            response = self.output_queue.get(timeout=60)
            # Update history
            history.append((message, response))
            return "", history  # Return empty string for input and updated history
        except queue.Empty:
            print("Timeout waiting for response")  # Debug print
            return "", history + [(message, "I'm sorry, I'm having trouble responding right now. Please try again.")]

    def run_healthg(self):
        def custom_input(*args):
            print("Waiting for input...")  # Debug print
            msg = self.input_queue.get()
            return msg

        def custom_print(message):
            self.output_queue.put(message)

        # Call the main function from HealthG with custom input and output functions
        healthg_main(custom_input=custom_input, custom_print=custom_print)

    def start_new_chat(self):
        # Clear the messages for the user
        self.simple_chat_store.delete_messages(self.user_id)
        self.message_index = 0
        self.chat_memory.reset()

        # Persist the empty chat store
        self.simple_chat_store.persist("data/chat_storage.json")

        # Reset past_messages
        self.past_messages = []
        self.input_queue.empty()
        self.output_queue.empty()

        return []

    def launch(self):
        with gr.Blocks(theme="soft", fill_height=True) as iface:
            gr.Markdown("# HealthG: Your Personal Health Assistant")
            gr.Markdown(
                "Welcome to HealthG! I'm here to assist you with health-related questions and advice. How can I help "
                "you today?")
            chatbot = gr.Chatbot(value=self.past_messages, height=600)
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
