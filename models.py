# models.py

class ChatRequest:
    def __init__(self, message):
        self.message = message

class ChatResponse:
    def __init__(self, response: str):
        self.response = response

    def to_dict(self):
        return {
            "response": self.response
        }