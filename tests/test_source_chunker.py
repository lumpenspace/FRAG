import pytest

from typing import List

from frag.embeddings.source_chunker import SourceChunker, ChunkingSettings
from frag.embeddings.embeddings_model import EmbeddingModel

class MockEmbeddingModel(EmbeddingModel):
    def tokenize(self, text:str):
        return list(range(len(text.split(' '))))
    def embed(self, text: str) -> List[float]:
        return [float(i) for i in range(len(text.split(' ')))]
    def decode(self, tokens: List[int]) -> str:
        return ' '.join(['Word{}'.format(i+1) for i in tokens])

from frag.embeddings.embeddings_model import EmbeddingModel

@pytest.fixture
def chunker():
    embedding_model = MockEmbeddingModel(dimensions=100, max_tokens=50)
    settings = ChunkingSettings()
    return SourceChunker(settings=settings, embedding_model=embedding_model)

@pytest.fixture
def chunker_with_sentences_preserved(chunker):
    chunker.settings.preserve_sentences = True
    return chunker

def test_chunk_text_preserve_paragraphs(chunker):
    text = "Paragraph 1.\n\nParagraph 2."
    chunker.settings.preserve_paragraphs = True
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1

def test_chunk_text_preserve_sentences(chunker):
    text = "Sentence 1. Sentence 2."
    chunker.settings.preserve_sentences = True
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1

def test_chunk_no_preserve(chunker):
    text = "Word1 Word2 Word3"
    chunker.settings.preserve_paragraphs = False
    chunker.settings.preserve_sentences = False
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1

def test_chunk_text_max_tokens(chunker):
    text = "A very long sentence that exceeds the max token limit."
    chunker.settings.preserve_sentences = True
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0

def test_chunk_text_long_paragraphs(chunker):
    long_paragraph = "This is a very long paragraph. " * 100
    text = f"{long_paragraph}\n\n{long_paragraph}"
    chunker.settings.preserve_paragraphs = True
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 2  # Expecting more than 2 chunks, as the paragraphs are long

def test_chunk_text_long_sentences(chunker):
    long_sentence = "This is a very long sentence. " * 100
    text = f"{long_sentence} {long_sentence}"
    chunker.settings.preserve_sentences = True
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 2  # Expecting more than 2 chunks, as the sentences are long

def test_chunk_text_mixed_settings(chunker):
    text = "Paragraph 1. Sentence 1. Sentence 2.\n\nParagraph 2. Sentence 1. Sentence 2."
    chunker.settings.preserve_paragraphs = True
    chunker.settings.preserve_sentences = True
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1  # Expecting 1 chunk, as the text fits within the token limit
    assert chunks[0]['text'] == "Word1 Word2 Word3 Word4 Word5 Word6 Word1 Word2 Word3 Word4 Word5 Word6"

def test_chunk_text_with_buffers(chunker):
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12" * 10
    chunker.settings.buffer_before = 2
    chunker.settings.buffer_after = 2
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0]['text'] starts
    assert chunks[0]['before_buffer'] == ""
    assert chunks[0]['after_buffer'] == "Word47 Word48"
    assert chunks[1]['before_buffer'] == "Word45 Word46"
    assert chunks[1]['after_buffer'] == "Word93 Word94"

def test_chunk_text_with_large_buffers(chunker):
    text = "Word1 Word2 Word3 Word4 Word5"
    chunker.settings.buffer_before = 3
    chunker.settings.buffer_after = 3
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0]['text'] == "Word1 Word2 Word3 Word4 Word5"
    assert chunks[0]['before_buffer'] == ""
    assert chunks[0]['after_buffer'] == ""

def test_chunk_text_with_zero_buffers(chunker):
    text = "Word1 Word2 Word3 Word4 Word5"
    chunker.settings.buffer_before = 0
    chunker.settings.buffer_after = 0
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0]['text'] == "Word1 Word2 Word3 Word4 Word5"
    assert chunks[0]['before_buffer'] == ""
    assert chunks[0]['after_buffer'] == ""