from typing import List

from frag.embeddings.apis import EmbedAPI

class EmbedAPITest(EmbedAPI):

    name: str = "EmbedTest"
    max_tokens: int = 50
    dimensions: int = 100

    def encode(self, text:str):
        return list(range(len(text.split(' '))))

    def embed(self, input: List[str]) -> List[List[float]]:
        return [[float(i) for i in range(self.dimensions)] for _ in input]

    def decode(self, tokens: List[int]) -> str:
        return ' '.join(['Word{}'.format(i+1) for i in tokens])


