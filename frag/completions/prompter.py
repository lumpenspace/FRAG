from logging import Logger, getLogger
from typing import List, Any
from litellm import Choices
from frag.settings import LLMSettings
from frag.typedefs import Message
from .summarizer_bot import SummarizerBot
from .interface_bot import InterfaceBot


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

    def respond(self, messages: List[Message], **kwargs: Any) -> str:
        """
        Respond to a message history.
        """
        try:
            responses: List[Choices] = self.interface.run(messages, **kwargs).choices
            if responses and responses[0] and responses[0].get:
                return str(responses[0].get("content", ""))
            return ""
        except Exception as e:
            self.logger.error("Error in responding: %s", e)
            raise

    def summarise(self, messages, **kwargs):
        try:
            return self.summarizer.run(messages, **kwargs)
        except Exception as e:
            self.logger.error("Error in summarising: %s", e)
            raise
