import os
import sys
import logging

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
)

from llama_index import download_loader
download_loader("GithubRepositoryReader")
from llama_hub.github_repo import GithubRepositoryReader, GithubClient

from llama_index.vector_stores.qdrant import QdrantVectorStore

# To connect to the same event-loop,
# allows async events to run on notebook
import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# for local testing
"""
client = qdrant_client.QdrantClient(
    # location=":memory:"
    # Async upsertion does not work
    # on 'memory' location and requires
    # Qdrant to be deployed somewhere.
    url="http://localhost:6334",
    prefer_grpc=True,
    # set API KEY for Qdrant Cloud
    #api_key=os.get(QDRANT_API_KEY"),
)
"""
# simple document example for testing
"""
print("loading documents...")
documents = SimpleDirectoryReader("./chainlit/docs").load_data()
"""

# Qdrant cloud

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

from qdrant_client import QdrantClient

client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
)


# TODO: You will need to set a Github token
# Read backstage github repo
# https://github.com/backstage/backstage/tree/master/docs
# Load all markdown files from docs directory
github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
loader = GithubRepositoryReader(
    github_client,
    owner =                  "backstage",
    repo =                   "backstage",
    filter_directories =     (["docs"], GithubRepositoryReader.FilterType.INCLUDE),
    filter_file_extensions = ([".md"], GithubRepositoryReader.FilterType.INCLUDE),
    verbose =                True,
    concurrent_requests =    10,
)
documents = loader.load_data(branch="master")

from llama_index.embeddings import TogetherEmbedding
service_context = ServiceContext.from_defaults(
    embed_model=TogetherEmbedding("togethercomputer/m2-bert-80M-8k-retrieval"),
)

vector_store = QdrantVectorStore(    
    client=client, collection_name="backstage", service_context=service_context
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(

    documents=documents,
    storage_context=storage_context,
    service_context=service_context,
    # FIXME asyncio throws error
    #use_async=True,
    show_progress=True,
)

# FIXME add refresh example