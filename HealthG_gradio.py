import gradio as gr
import queue
import threading
from HealthG import HealthG
from config import GRADIO_THEME, CHATBOT_HEIGHT


class HealthGGradio:
    def __init__(self):
        self.user_id = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.health_g = None
        self.chatbot = []

        self.healthg_thread = threading.Thread(target=self.run_healthg)
        self.healthg_thread.start()

    def set_user_id(self, user_id):
        self.user_id = user_id
        self.health_g = HealthG(user_id)  # Initialize HealthG here
        self.chatbot = self.health_g.get_past_messages()
        return self.user_id, self.chatbot

    def chat(self, message, history):
        if self.health_g is None:
            return "", history + [("", "Please set a user ID first.")]
        self.input_queue.put(message)
        try:
            response = self.output_queue.get(timeout=60)
            history.append((message, response))
            return "", history
        except queue.Empty:
            return "", history + [(message, "I'm sorry, I'm having trouble responding right now. Please try again.")]

    def run_healthg(self):
        def custom_input():
            return self.input_queue.get()

        def custom_print(message):
            self.output_queue.put(message)

        while True:
            user_query = custom_input()
            if user_query.lower() == 'e':
                break
            if self.health_g is None:
                custom_print("Please set a user ID first.")
            else:
                response = self.health_g.chat(user_query)
                custom_print(response)

    def start_new_chat(self):
        self.health_g.reset_chat()
        return []

    def launch(self):
        with gr.Blocks(theme=GRADIO_THEME, fill_height=True) as iface:
            gr.Markdown("# HealthG: Your Personal Health Assistant")
            gr.Markdown(
                "Welcome to HealthG! I'm here to assist you with health-related questions and advice. How can I "
                "help you today?")
            with gr.Group() as user_id_group:
                user_id = gr.Textbox(placeholder="Enter Username Here...", label="Username",
                                     info="Enter your username here so I know who you are.", interactive=True,
                                     autofocus=True)
            with gr.Group(visible=False) as main_interface:
                chatbot = gr.Chatbot(height=CHATBOT_HEIGHT, label="HealthG", container=False)
                msg = gr.Textbox(label="HealthG", container=False, autoscroll=True, autofocus=True,
                                 placeholder="Type your health-related question here...")
                with gr.Row():
                    gr.ClearButton([msg, chatbot], value="Clear Chat Window")
                    new_chat = gr.Button("Start New Chat")

                gr.Examples(
                    examples=[
                        "What are some tips for maintaining a healthy diet?",
                        "How can I improve my sleep quality?",
                        "What are the benefits of regular exercise?",
                        "How can I manage stress effectively?"
                    ],
                    inputs=msg
                )

            def show_interface(si_user_id):
                si_user_id, chat_history = self.set_user_id(si_user_id)
                if si_user_id.strip():  # Check if user_id is not empty
                    return (gr.Group(visible=False),  # Hide user ID group
                            gr.Group(visible=True),  # Show main interface
                            chat_history)
                return (gr.Group(visible=True),  # Keep user ID group visible
                        gr.Group(visible=False),  # Keep main interface hidden
                        [])

            user_id.submit(self.set_user_id, inputs=user_id)
            user_id.submit(show_interface, inputs=user_id, outputs=[user_id_group, main_interface, chatbot])
            msg.submit(self.chat, [msg, chatbot], [msg, chatbot])
            new_chat.click(self.start_new_chat, outputs=[chatbot])

        iface.launch(inbrowser=True, share=True)
