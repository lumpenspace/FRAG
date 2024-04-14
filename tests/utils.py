"""
This module contains the EmbedAPITest class which extends the EmbedAPI class for testing purposes.
It provides methods to encode text into a list of integers, embed a list of strings into a list of float lists,
and decode a list of integers back into a string.
"""

from typing import List

from frag.embeddings.apis import EmbedAPI

class EmbedAPITest(EmbedAPI):
    """
    A test implementation of the EmbedAPI for testing embedding functionalities.
    
    Methods:
        encode: Converts a string into a list of integers based on word count.
        embed: Embeds a list of strings into a list of lists of floats.
        decode: Converts a list of integers back into a concatenated string.
    """
    
    name: str = "EmbedTest"
    max_tokens: int = 50
    dimensions: int = 100

    def encode(self, text:str):
        """
        Encodes the given text into a list of integers representing the word count.
        
        Args:
            text (str): The text to encode.
            
        Returns:
            List[int]: A list of integers representing the word count.
        """
        return list(range(len(text.split(' '))))

    def embed(self, input: List[str]) -> List[List[float]]:
        """
        Embeds the given list of strings into a list of lists of floats.
        
        Args:
            input (List[str]): The list of strings to embed.
            
        Returns:
            List[List[float]]: A list of lists of floats representing the embeddings.
        """
        return [[float(i) for i in range(self.dimensions)] for _ in input]

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes the given list of integers back into a concatenated string.
        
        Args:
            tokens (List[int]): The list of integers to decode.
            
        Returns:
            str: The concatenated string.
        """
        return ' '.join(['Word{}'.format(i+1) for i in tokens])


