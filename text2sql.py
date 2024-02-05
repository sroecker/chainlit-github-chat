import os
# enable logging to see generated SQL
import logging
import sys

#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    ServiceContext,
    SQLDatabase,    
)
from llama_index.indices.struct_store import (
    NLSQLTableQueryEngine,
    SQLTableRetrieverQueryEngine,
)

from sqlalchemy import (
    create_engine,
    text,
)


"""
engine = create_engine("duckdb:///:memory:")
with engine.connect() as conn:
    r = conn.execute(
            text("CREATE TABLE sales_data as SELECT * FROM read_parquet('git_commits.parquet');")
    )
    conn.commit()
"""


DUCKDB_TOKEN = os.getenv('DUCKDB_TOKEN')
# create duckdb engine and connect to MotherDuck
engine = create_engine(f"duckdb:///md:{DUCKDB_TOKEN}@my_db")

with engine.connect() as conn:
    r = conn.execute(
            text("select * from git_commits limit 5")
    )
    print(r.fetchall())


from llama_index import SQLDatabase
sql_database = SQLDatabase(engine, include_tables=["git_commits"])


"""
# set up connection to Ollama
from llama_index.llms import Ollama
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
print(f"Connecting to ollama server {OLLAMA_HOST}")
#my_llm = Ollama(model="duckdb-nsql", base_url="http://"+OLLAMA_HOST+":11434")
#my_llm = Ollama(model="mistral-openorca", base_url="http://"+OLLAMA_HOST+":11434")
my_llm = Ollama(model="starling-lm", base_url="http://"+OLLAMA_HOST+":11434")
"""

#from llama_index import ServiceContext
#service_context = ServiceContext.from_defaults(llm=my_llm, embed_model="local")


from llama_index.embeddings import TogetherEmbedding
from llama_index.llms import TogetherLLM

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
)

nsql_query_engine = NLSQLTableQueryEngine(sql_database = sql_database, service_context = service_context)

res = nsql_query_engine.query("Which author by name had the most commits")
print(res)

res = nsql_query_engine.query("What was the week with the most commits?")
print(res)
