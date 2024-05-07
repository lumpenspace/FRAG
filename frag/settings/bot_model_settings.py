"""
LLM settings, used for both the interface and summarizer bots.
"""

from typing import Any, Dict

from litellm.utils import get_model_info, get_supported_openai_params
from pydantic import model_validator
from pydantic_settings import BaseSettings


class BotModelSettings(BaseSettings):
    """
    LLM settings, used for both the interface and summarizer bots.
    """

    api: str = "gpt-3.5-turbo"
    bot: str

    def dump(self) -> dict[str, Any]:
        """
        return model settings
        """
        return self.model_dump(exclude_unset=True, exclude_none=True)

    def __getitem__(self, item: str) -> Any:
        return self.model_dump(exclude_unset=True, exclude_none=True)[item]

    @property
    def api_name(self) -> str:
        """
        Returns the API name for the model.
        """
        return self.model_dump(exclude_unset=True, exclude_none=True)["api"]

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Makes sure all params are supported by the model.
        """
        other_values: Dict[str, Any] = {
            k: v for k, v in values.items() if k not in ["api", "bot"]
        }
        # check if values are supported
        model_string: str = values.get("api", "gpt-3.5-turbo")

        try:
            info: Dict[str, Any] = get_model_info(model_string)
        except Exception:
            raise ValueError(
                f"Error getting model info for {model_string}.\n\
                List of supported models:\n\
                https://docs.litellm.ai/docs/providers\n"
            )
        supported_params: list[str] | None = get_supported_openai_params(
            model_string, info.get("litellm_provider", None)
        )
        if supported_params is None:
            raise ValueError("Model {model_string} is not supported by litellm.")
        for k, _ in other_values.items():
            if k not in supported_params:
                raise ValueError(
                    f"Unsupported parameter {k} for model {values.get('model', 'gpt-3.5-turbo')}"
                )
        return {
            **other_values,
            "api": model_string,
            "bot": values.get("bot"),
        }
