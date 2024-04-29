from pydantic import BaseModel
from openai.types.chat import (
    ChatCompletionMessage as Message,
    ChatCompletionAssistantMessageParam as AssistantMessage,
    ChatCompletionMessageParam as MessageParam,
    ChatCompletionUserMessageParam as UserMessage,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionRole as Role,
    ChatCompletionFunctionMessageParam as FunctionMessage,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase as CompletionParams,
)


class Note(BaseModel):
    id: str
    source: str
    title: str
    summary: str
    complete: bool


Role = Role
CompletionParams = CompletionParams
MessageParam = MessageParam
SystemMessage = SystemMessage
FunctionMessage = FunctionMessage
AssistantMessage = AssistantMessage
UserMessage = UserMessage
Message = Message
