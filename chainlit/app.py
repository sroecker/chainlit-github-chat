import os

from llama_index.response.schema import Response, StreamingResponse
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
)

from llama_index.embeddings import TogetherEmbedding
from llama_index.llms import TogetherLLM

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

import chainlit as cl

QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')

STREAMING = True

# Provide a template following the LLM's original chat template.
#def completion_to_prompt(completion: str) -> str:
#  return f"<s>[INST] {completion} [/INST] </s>\n"



@cl.on_chat_start
async def factory():
    my_llm = TogetherLLM(
        model="teknium/OpenHermes-2p5-Mistral-7B",
        temperature=0.0,
        max_tokens=256,
        top_p=0.7,
        top_k=50,
        # stop=...,
        # repetition_penalty=...,
        is_chat_model=True,
        #completion_to_prompt=completion_to_prompt
    )
        
    service_context = ServiceContext.from_defaults(
        llm=my_llm,
        embed_model=TogetherEmbedding("togethercomputer/m2-bert-80M-8k-retrieval"),
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
    )
    vector_store = QdrantVectorStore(client=client, collection_name="backstage", prefer_grpc=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = []

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )

    query_engine = index.as_query_engine(
        service_context=service_context,
        streaming=STREAMING,
    )

    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")
    await response_message.send()

    if isinstance(response, Response):
        response_message.content = str(response)
        await response_message.update()
    elif isinstance(response, StreamingResponse):
        gen = response.response_gen
        for token in gen:
            await response_message.stream_token(token=token)

        if response.response_txt:
            response_message.content = response.response_txt

        await response_message.update()

