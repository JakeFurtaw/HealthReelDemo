import gradio as gr
from config import GRADIO_THEME, CHATBOT_HEIGHT
from gr_utils import HealthBotGradio as GRUtils

grUtils = GRUtils()

with gr.Blocks(theme=GRADIO_THEME, fill_height=True, fill_width=True, title="Health Bot") as demo:
    gr.Markdown("# Health Bot: Your Personal Health Assistant")
    gr.Markdown("Welcome to Health Bot! I'm here to assist you with health-related questions and advice.")

    with gr.Group() as user_id_group:
        user_id = gr.Textbox(placeholder="Enter Username Here...", label="Username",
                             info="Enter your username here so I know who you are.", interactive=True,
                             autofocus=True)

    with gr.Group(visible=False) as main_interface:
        chatbot = gr.Chatbot(height=CHATBOT_HEIGHT, label="Health Bot", container=False)
        msg = gr.Textbox(container=False, autoscroll=True, autofocus=True,
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
        si_user_id, chat_history = grUtils.set_user_id(si_user_id)
        return (gr.Group(visible=not bool(si_user_id.strip())),
                gr.Group(visible=bool(si_user_id.strip())),
                chat_history if si_user_id.strip() else [])


    user_id.submit(show_interface, inputs=user_id, outputs=[user_id_group, main_interface, chatbot])
    msg.submit(grUtils.chat, [msg, chatbot], [msg, chatbot])
    new_chat.click(grUtils.start_new_chat, outputs=[chatbot])

demo.launch(inbrowser=True, share=True)
