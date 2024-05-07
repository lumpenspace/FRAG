"""
Interface bot. This is the bot that will directly interact with the user.

To personalise the template, extract (`frag template extract`) and modify them from:
- .frag/templates/system.j2
- .frag/templates/user.j2
"""

from typing import List, Literal

import jinja2
from litellm import completion, ModelResponse

from frag.typedefs import MessageParam, Note
from frag.settings import BotModelSettings
from frag.console import error_console

from .base_bot import BaseBot


class InterfaceBot(BaseBot):
    """
    Interface bot. This is the bot that will directly interact with the user.

    It is called by `prompter.py`.
    """

    client_type = "interface"
    settings: BotModelSettings
    system_template: jinja2.Template
    user_template: jinja2.Template

    def __init__(self, settings: BotModelSettings, template_dir: str) -> None:
        super().__init__(settings, template_dir=template_dir)

    def run(self, messages: List[MessageParam], notes: List[Note]) -> ModelResponse:
        try:
            rendered_messages: List[MessageParam] = self._render(messages, notes=notes)

            result = ModelResponse(
                completion(
                    messages=rendered_messages,
                    model=self.settings.api,
                    stream=False,
                    **self.settings.model_dump(exclude=set(["llm"])),
                )
            )
            return result
        except jinja2.TemplateError as te:
            error_console("Template rendering error: %s", te)
            raise
        except Exception as e:
            error_console("General error during completion: %s", e)
            raise

    def _render(self, messages: List[MessageParam], **kwargs) -> List[MessageParam]:
        try:
            last_message = messages[-1]
            return [
                self._render_message(messages=messages[:-1], role="system", **kwargs),
                *messages[:-1],
                self._render_message(
                    messages=last_message.get("content"), role="user", **kwargs
                ),
            ]
        except IndexError as ie:
            error_console(
                "Rendering error - possibly due to empty messages list: %s", ie
            )
            raise
        except Exception as e:
            error_console("General rendering error: %s", e)
            raise
