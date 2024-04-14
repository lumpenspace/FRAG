from frag.embeddings.apis import  OpenAIEmbedAPI, EmbedAPI, SentenceEmbedAPI

def get_embed_api(embed_api: EmbedAPI|str) -> EmbedAPI:
    if not embed_api:
            raise ValueError("Embedding model and chunking settings must be provided")
    if isinstance(embed_api, EmbedAPI):
        return embed_api
    if isinstance(embed_api, str):
        if embed_api.startswith('oai:'):
            return OpenAIEmbedAPI(name=embed_api.replace('oai:', ''))
        return SentenceEmbedAPI(name=embed_api)
    if issubclass(embed_api, EmbedAPI):
        return embed_api()

