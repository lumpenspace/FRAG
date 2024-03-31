from typing import List
from frag.embeddings.embedding_model import EmbeddingModel

class MyEmbeddingModel(EmbeddingModel):
    max_tokens: int = 50
    dimensions: int = 100

    def tokenize(self, text:str):
        return list(range(len(text.split(' '))))

    def embed(self, text: str) -> List[float]:
        return [float(i) for i in range(len(text.split(' ')))]

    def decode(self, tokens: List[int]) -> str:
        return ' '.join(['Word{}'.format(i+1) for i in tokens])
    