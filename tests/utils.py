from typing import List, Type

from frag.embeddings.embed_api import EmbedAPI

class EmbedAPITest(EmbedAPI):
    __test__ = False
    name: str = "TestEmbedAPI"
    max_tokens: int = 50
    dimensions: int = 100

    def tokenize(self, text:str):
        return list(range(len(text.split(' '))))

    def embed(self, text: str) -> List[float]:
        return [float(i) for i in range(len(text.split(' ')))]

    def decode(self, tokens: List[int]) -> str:
        return ' '.join(['Word{}'.format(i+1) for i in tokens])


