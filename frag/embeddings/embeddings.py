import os
import hashlib

from typing import List, Type
from pydantic import BaseModel, Field, model_validator

from chromadb import ClientAPI, Collection, Client

from .embeddings_metadata import ChunkMetadata, Metadata
from .source_chunker import SourceChunker, ChunkingSettings
from .embeddings_model import EmbeddingModel, OpenAiEmbeddingModel, openai_embedding_models

class Embeddings(BaseModel):
    """
    A class for handling embeddings using various models.
    """
    api_key: str = Field("", description="API key for the embedding model")
    database_path: str = Field("./chroma_db", description="Path to the Chroma database")
    chunking_settings: ChunkingSettings = Field(..., description="Settings for chunking the text")
    chunker: SourceChunker = Field(..., description="Chunker for the embeddings")
    chroma_client: Type[ClientAPI]  = Field(..., description="Chroma client for the database")
    embedding_model: EmbeddingModel = Field(..., description="Embedding model")

    @model_validator(mode='before')
    def validate_chunking_settings(cls, values):
        chunking_settings = values.get('chunking_settings')
        if not isinstance(chunking_settings, ChunkingSettings):
            raise ValueError(f"Expected 'chunking_settings' to be of type 'ChunkingSettings', got {type(chunking_settings)}")
        return values

    @model_validator(mode='before')
    def validate_embedding_model(cls, values):
        embedding_model = values.get('embedding_model')
        api_key = values.get('api_key')

        if isinstance(embedding_model, str):
            if embedding_model not in openai_embedding_models:
                raise ValueError(f"Unsupported embedding model: {embedding_model}")
            values['embedding_model'] = OpenAiEmbeddingModel(name=embedding_model, api_key=api_key)
        elif isinstance(embedding_model, EmbeddingModel):
            values['embedding_model'] = embedding_model
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")

        return values

    @model_validator(mode='before')
    def validate_chunker(cls, values):
        chunking_settings = values.get('chunking_settings')
        embedding_model = values.get('embedding_model')

        if chunking_settings and embedding_model:
            values['chunker'] = SourceChunker(chunking_settings=chunking_settings, embedding_model=embedding_model)

        return values

    @model_validator(mode='before')
    def validate_chroma_client(cls, values):
        database_path = values.get('database_path')

        # check the path's accessibility
        if database_path and os.path.exists(database_path):
            values['chroma_client'] = Client(database_path=database_path)
        else:
            raise ValueError(f"Database path {database_path} does not exist")

        return values

    @property
    def name(self) -> str:
        """Returns the name of the embedding model."""
        return self.embedding_model.name


    def fetch_embedding(self, text: str) -> List[float]:
        """Returns the embedding vector for the given text."""
        return self.embedding_model.embed(text)
        
        
    def create_embeddings_for_document(self, text: str, metadata: Metadata):
        """
        Creates embeddings for a document and stores them in a Chroma database.
        """
        chunks = self.chunker.chunk_text(text)
        parts = len(chunks)
        for index, chunk in enumerate(chunks):
            # add a "part" attribute to the metadata
            chunk_metadata = ChunkMetadata(**metadata.model_dump(), part=index + 1, parts=parts)
            self.fetch_and_store_embedding(chunk['text'], metadata=chunk_metadata)


    def fetch_and_store_embedding(self, text: str, metadata: ChunkMetadata):
        """
        Stores the embedding in a Chroma database and returns it.

        Parameters:
            text: The text of the document
            metadata: The metadata associated with the embedding.

        Returns:
            The embedding vector.
        """
        try:
            collection:Collection = self.chroma_client.get_or_create_collection(metadata.name)
            # add a hash of the text
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            id = metadata.title.lower().replace(" ", "_")+str(metadata.part)+"-"+str(metadata.parts)+"-"+text_hash

            collection_result = collection.get(ids=id)
            
            if collection_result and collection_result[0]:
                print("Embedding found in db")
                return collection_result[0]

            print("getting embeddings")
            # Get the embedding for the text
            embedding = self.fetch_embedding(text)
            
            # Store the text and its embedding in Chroma
            collection.add(ids=id, embeddings=embedding, documents=text, metadatas=metadata.model_dump())

            return embedding
        except Exception as e:
            print(f"Error storing embedding: {e}")
            # Log the error or handle it appropriately
            return None
