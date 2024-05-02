"""
LLM settings, used for both the interface and summarizer bots.

In .fragrc, the settings are under `bots` - default ones direct descendants, and bot specifics
under the bot name. For example, here:

```yaml
bots:
    api: gpt3-turbo
    max_tokens: 1000
    top_p: 1.0
    interface:
        model: oai:gpt4
    summarizer:
        temperature: 0.5
```

... the `model` key is the default model for all bots, and the `interface` and `summarizer` keys
are bot specific settings. If a bot specific setting is not present, the default one is used.
"""

from typing import Dict, Any

from .llm_model_settings import LLMModelSettings


class LLMSettings(LLMModelSettings):
    """
    LLM settings, used for both the interface and summarizer bots.
    """

    interface_bot: LLMModelSettings
    summarizer_bot: LLMModelSettings

    @classmethod
    def from_dict(
        cls,
        interface_bot: Dict[str, Any] | None = None,
        summarizer_bot: Dict[str, Any] | None = None,
        **kwargs: Dict[str, Any]
    ) -> "LLMSettings":
        """
        Create an LLMSettings object from a dictionary.
        """
        interface: Dict[str, Any] = interface_bot or dict()
        summarizer: Dict[str, Any] = summarizer_bot or dict()
        default_settings: Dict[str, Any] = kwargs
        return cls(
            interface_bot=LLMModelSettings(**default_settings, **interface),
            summarizer_bot=LLMModelSettings(**default_settings, **summarizer),
        )
