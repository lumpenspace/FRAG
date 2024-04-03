import pytest

from typing import Any, List

from frag import ChunkingSettings
from .utils import EmbeddingModelTest
from frag.embeddings.write.source_chunker import SourceChunker, ChunkingSettings

@pytest.fixture
def chunker():
    return SourceChunker(settings=ChunkingSettings(), embedding_model=EmbeddingModelTest())

def chunker_validatable_dict(chunker:SourceChunker, **kwargs):
    return {"settings": ChunkingSettings(**{**chunker.settings.model_dump().copy(), **kwargs}), "embedding_model": EmbeddingModelTest()}

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
    assert chunks[0].text == "Word1 Word2 Word3 Word4 Word5 Word6 Word1 Word2 Word3 Word4 Word5 Word6"

def test_chunk_text_with_buffers(chunker):
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12" * 10
    chunker.settings.buffer_before = 2
    chunker.settings.buffer_after = 2
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0].text == "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12 Word13 Word14 Word15 Word16 Word17 Word18 Word19 Word20 Word21 Word22 Word23 Word24 Word25 Word26 Word27 Word28 Word29 Word30 Word31 Word32 Word33 Word34 Word35 Word36 Word37 Word38 Word39 Word40 Word41 Word42 Word43 Word44 Word45 Word46"
    assert chunks[0].before == ""
    assert chunks[0].after == "Word47 Word48"
    assert chunks[1].before == "Word45 Word46"
    assert chunks[1].after == "Word93 Word94"

def test_chunk_text_with_large_buffers(chunker):
    text = "Word1 Word2 Word3 Word4 Word5"
    chunker.settings.buffer_before = 3
    chunker.settings.buffer_after = 3
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == "Word1 Word2 Word3 Word4 Word5"
    assert chunks[0].before == ""
    assert chunks[0].after == ""

def test_chunk_text_with_zero_buffers(chunker):
    text = "Word1 Word2 Word3 Word4 Word5"
    chunker.settings.buffer_before = 0
    chunker.settings.buffer_after = 0
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == "Word1 Word2 Word3 Word4 Word5"
    assert chunks[0].before == ""
    assert chunks[0].after == ""

def test_validate_settings_passes_with_positive_buffer_and_max_tokens(chunker):
    try:
        chunker.validate_settings(chunker_validatable_dict(chunker, settings=ChunkingSettings(max_tokens=50)))
    except ValueError:
        pytest.fail("validate_settings() raised ValueError unexpectedly!")