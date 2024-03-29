from typing import List
from pydantic import BaseModel, Field
import os
import tiktoken
from openai import OpenAI

class EmbeddingModel(BaseModel):
    """
    A base class for embedding models.

    Attributes:
        dimensions: The dimensionality of the embedding vectors.
        max_tokens: The maximum number of tokens to consider for embedding.
    """
    dimensions: int = Field(..., description="The dimensionality of the embedding vectors")
    max_tokens: int = Field(..., description="The maximum number of tokens to consider for embedding")

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("Subclasses must implement the tokenize method")

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement the embed method")

openai_embedding_models = ["text-embedding-ada-002", "text-embedding-large", "text-embedding-small"]

class OpenAiEmbeddingModel(EmbeddingModel):
    """
    A class to interact with OpenAI's embedding models.

    Attributes:
        model_name: The name of the OpenAI embedding model.
        api_key: The API key for the OpenAI client.
        tokenizer_name: The name of the tokenizer to use.
    """
    model_name: str = Field("text-embedding-small", description="The name of the OpenAI embedding model")
    api_key: str = Field(..., description="The API key for the OpenAI client")
    tokenizer_name: str = Field("cl100k_base", description="The name of the tokenizer to use")

    def __init__(self, **data):
        super().__init__(**data)

        if self.model_name not in openai_embedding_models:
            raise ValueError(f"Unsupported OpenAI embedding model: {self.model_name}")

        self.openai_client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.get_encoding(self.tokenizer_name)

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def embed(self, text: str) -> List[float]:
        try:
            embedding_object = self.openai_client.embeddings.create(input=text, model=self.model_name)
        except Exception as e:
            raise ValueError(f"OpenAiEmbeddingsModel: error embedding text: {e}")

        embedding_vector = embedding_object.data[0].embedding

        return embedding_vector