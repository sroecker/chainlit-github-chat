import os

from llama_index.response.schema import Response, StreamingResponse
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.query_engine import SQLJoinQueryEngine, RetrieverQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools import ToolMetadata
from llama_index.indices.vector_store import VectorIndexAutoRetriever

from sqlalchemy import (
    create_engine,
)


from llama_index.embeddings import TogetherEmbedding
from llama_index.llms import TogetherLLM

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

import chainlit as cl

QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
DUCKDB_TOKEN = os.getenv('DUCKDB_TOKEN')

STREAMING = True

# Provide a template following the LLM's original chat template.
#def completion_to_prompt(completion: str) -> str:
#  return f"<s>[INST] {completion} [/INST] </s>\n"



@cl.on_chat_start
async def factory():


    
    # create duckdb engine and connect to MotherDuck
    engine = create_engine(f"duckdb:///md:{DUCKDB_TOKEN}@my_db")
    from llama_index import SQLDatabase
    sql_database = SQLDatabase(engine, include_tables=["git_commits"])

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
    vector_store = QdrantVectorStore(
        client=client, collection_name="backstage", prefer_grpc=True
    )
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )

    v_engine = index.as_query_engine(
        service_context=service_context,
        streaming=STREAMING,
    )

    nlsql_query_engine = NLSQLTableQueryEngine(
        sql_database = sql_database,
        service_context = service_context
    )

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=nlsql_query_engine,
        description=(
            "Useful for translating a natural language query into a SQL query over"
            "a table containing the git commit history of the Backstage project"
            "with the fields: author_name, message, committer_when"
        ),
    )
    v_engine_tool = QueryEngineTool.from_defaults(
        query_engine=v_engine,
        description=(
            f"Useful for answering semantic questions Backstage documentation"
        ),
    )
    
    # join query engine
    query_engine = SQLJoinQueryEngine(
        sql_tool, v_engine_tool, service_context=service_context
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

