from typing import List, Any
from litellm import Choices
from frag.settings import BotsSettings
from frag.typedefs import Message
from .summarizer_bot import SummarizerBot
from .interface_bot import InterfaceBot
from frag.utils.console import error_console


class Prompter:
    """
    Main handler of prompts and responses.

    """

    def __init__(
        self, settings: BotsSettings, summarizer: SummarizerBot, interface: InterfaceBot
    ) -> None:
        self.settings: BotsSettings = settings
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
            error_console.log("Error in responding: %s", e)
            raise

    def summarise(self, messages: List[Message], **kwargs: Any):
        try:
            return self.summarizer.run(messages, **kwargs)
        except Exception as e:
            error_console.log("Error in summarising: %s", e)
            raise
