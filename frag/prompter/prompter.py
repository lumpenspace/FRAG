from logging import Logger, getLogger
from typing import List
from litellm import Choices
from frag.typedefs import LLMSettings
from .summarizer import SummarizerBot
from .interface import InterfaceBot


class Prompter:
    """
    Main handler of prompts and responses.

    """

    def __init__(
        self, settings: LLMSettings, summarizer: SummarizerBot, interface: InterfaceBot
    ) -> None:
        self.logger: Logger = getLogger(__name__)
        self.settings: LLMSettings = settings
        self.summarizer: SummarizerBot = summarizer
        self.interface: InterfaceBot = interface

    def respond(self, messages, **kwargs) -> str:
        """
        Respond to a message history.
        """
        try:
            response: List[Choices] = self.interface.run(messages, **kwargs).choices
            return response[0].content if response else ""
        except Exception as e:
            self.logger.error("Error in responding: %s", e)
            raise

    def summarise(self, messages, **kwargs):
        try:
            return self.summarizer.run(messages, **kwargs)
        except Exception as e:
            self.logger.error("Error in summarising: %s", e)
            raise
