from frag.types import Settings
from .summarizer import Summarizer
from .interface import Interface

class Prompter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.summarizer = Summarizer(settings=self.settings.summarizer)
        self.interface = Interface(settings=self.settings.interface)

    def respond(self, messages, **kwargs):
        return self.interface.run(messages, **kwargs)

    def summarise(self, messages, **kwargs):
        return self.summarizer.run(messages, **kwargs)
