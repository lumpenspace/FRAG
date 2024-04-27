import logging

from frag.types import Settings
from .summarizer import Summarizer
from .interface import Interface


class Prompter:
    def __init__(
        self, settings: Settings, summarizer: Summarizer, interface: Interface
    ):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.summarizer = summarizer
        self.interface = interface

    def respond(self, messages, **kwargs):
        try:
            return self.interface.run(messages, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in responding: {e}")
            raise

    def summarise(self, messages, **kwargs):
        try:
            return self.summarizer.run(messages, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in summarising: {e}")
            raise
