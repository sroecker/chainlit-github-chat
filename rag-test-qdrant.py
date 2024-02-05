import os
import sys
import logging

from llama_index import SimpleDirectoryReader, StorageContext, ServiceContext, VectorStoreIndex
from llama_index.embeddings import TogetherEmbedding
from llama_index.llms import TogetherLLM

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

from qdrant_client import QdrantClient
client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
)


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


vector_store = QdrantVectorStore(client=client, collection_name="backstage", prefer_grpc=True)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

STREAMING = True
query_engine = index.as_query_engine(
    service_context=service_context,
    streaming=STREAMING,
)

queries = [
    "Is there a manged version of Backstage?",
    "How can I write a software template?",
]

for query_text in queries:
    response = query_engine.query(query_text)
    print(response)