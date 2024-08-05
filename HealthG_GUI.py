import gradio as gr
import queue
import threading
from HealthG import main as healthg_main


class HealthGradio:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.chat_history = []
        self.healthg_running = True

        # Start HealthG in a separate thread
        self.healthg_thread = threading.Thread(target=self.run_healthg)
        self.healthg_thread.start()

    def chat(self, message, history):
        print(f"Received message: {message}")  # Debug print
        self.input_queue.put(message)

        # Wait for the response from HealthG with a timeout
        try:
            response = self.output_queue.get(timeout=10)
            print(f"Received response: {response}")  # Debug print
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
        iface = gr.ChatInterface(
            self.chat,
            chatbot=gr.Chatbot(height=400),
            textbox=gr.Textbox(placeholder="Type your health-related question here...", container=False),
            title="HealthG: Your Personal Health Assistant",
            description="Welcome to HealthG! I'm here to assist you with health-related questions and advice. How can "
                        "I help you today?",
            theme="soft",
            examples=[
                "What are some tips for maintaining a healthy diet?",
                "How can I improve my sleep quality?",
                "What are the benefits of regular exercise?",
                "How can I manage stress effectively?"
            ],
            retry_btn=None,
            undo_btn="Delete Last",
            clear_btn="Clear",
        )

        iface.launch(share=True)


if __name__ == "__main__":
    app = HealthGradio()
    app.launch()
