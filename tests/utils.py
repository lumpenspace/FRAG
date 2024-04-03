from typing import List, Type

from frag.embeddings.embedding_model import EmbeddingModel

class EmbeddingModelTest(EmbeddingModel):
    __test__ = False
    name: str = "TestEmbeddingModel"
    max_tokens: int = 50
    dimensions: int = 100

    def tokenize(self, text:str):
        return list(range(len(text.split(' '))))

    def embed(self, text: str) -> List[float]:
        return [float(i) for i in range(len(text.split(' ')))]

    def decode(self, tokens: List[int]) -> str:
        return ' '.join(['Word{}'.format(i+1) for i in tokens])


