from frag.settings import Settings
from .summarizer import Summarizer

class Prompter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.summarizer = Summarizer(settings=self.settings.summarizer)

    def summarise(self, messages, **kwargs):
        return self.summarizer.run(messages, **kwargs)
