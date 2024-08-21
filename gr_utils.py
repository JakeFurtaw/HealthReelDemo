from chat import Chat
from models import load_embedding_model, load_llm


class HealthBotGradio:
    def __init__(self):
        self.user_id = None
        self.health_g = None
        self.embed_model = load_embedding_model()
        self.llm = load_llm()

    def set_user_id(self, user_id):
        self.user_id = user_id
        self.health_g = Chat(user_id, self.embed_model, self.llm)
        return self.user_id, self.health_g.get_past_messages()

    def chat(self, message, history):
        if self.health_g is None:
            return "", history + [("", "Please set a user ID first.")]
        response = self.health_g.chat(message)
        history.append((message, response))
        return "", history

    def start_new_chat(self):
        if self.health_g:
            self.health_g.reset_chat()
        return []