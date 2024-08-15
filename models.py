# models.py

class ChatRequest:
    def __init__(self, message):
        self.message = message

class ChatResponse:
    def __init__(self, response):
        self.response = response