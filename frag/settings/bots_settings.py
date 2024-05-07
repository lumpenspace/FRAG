"""
LLM settings, used for both the interface and summarizer bots.

In .fragrc, the settings are under `bots` - default ones direct descendants, and bot specifics
under the bot name. For example, here:

```yaml
bots:
    api: gpt-3.5-turbo
    max_tokens: 1000
    top_p: 1.0
    interface:
        api: gpt-4-turbo
    summarizer:
        temperature: 0.5
```

... the `model` key is the default model for all bots, and the `interface` and `summarizer` keys
are bot specific settings. If a bot specific setting is not present, the default one is used.
"""

from typing import Dict, Any, Self
from frag.console import console
from .bot_model_settings import BotModelSettings
from pydantic_settings import BaseSettings


class BotsSettings(BaseSettings):
    """
    LLM settings, used for both the interface and summarizer bots.
    """

    interface_bot: BotModelSettings
    summarizer_bot: BotModelSettings

    @classmethod
    def from_dict(
        cls,
        bots_dict: Dict[str, Any],
    ) -> Self:
        """
        Create an LLMSettings object from a dictionary.
        """
        interface_bot_settings: Dict[str, Any] = bots_dict.pop("interface", {})
        summarizer_bot_settings: Dict[str, Any] = bots_dict.pop("summarizer", {})
        default_settings: Dict[str, Any] = bots_dict

        console.log(
            f"[b]BotsSettings[/b]\
            \n\tdefault: {default_settings}\
            \n\tinterface: {interface_bot_settings}\
            \n\tsummarizer: {summarizer_bot_settings}"
        )

        return cls(
            interface_bot=BotModelSettings(
                **default_settings, **interface_bot_settings, bot="interface"
            ),
            summarizer_bot=BotModelSettings(
                **default_settings, **summarizer_bot_settings, bot="summarizer"
            ),
        )
