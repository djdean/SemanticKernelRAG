from azure.search.documents.models import VectorizedQuery 
from Config import Config
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI 
from azure.search.documents import SearchClient
from typing import Callable, Any, Set

search_service_endpoint = "https://vectorsearchdj.search.windows.net"
search_service_key = "yMQEaTjqO36z14nI1hTJB4BKCyHorAuJnw4Z06cbypAzSeBiPD2i"
index_name = "syneos_poc"
aoai_endpoint = "https://eus2testdj.openai.azure.com/"
aoai_key = "13872f737d1342cd80c226743687553a"
aoai_api_version = "2025-01-01-preview"
embedding_model = "text-embedding-3-large"
def connect_to_search_index():
    credentials = AzureKeyCredential(search_service_key)
    search_client = SearchClient(endpoint=search_service_endpoint, 
                                    credential=credentials, index_name=index_name)
    return search_client
def generate_embeddings(data):
    
    client = AzureOpenAI(
        azure_endpoint = aoai_endpoint,
        api_key=aoai_key,
        api_version=aoai_api_version
    )
    return client.embeddings.create(input = [data], model=embedding_model).data[0].embedding
def do_vector_search(input_query:str ) -> str:
    """
    Perform a vector search using the input query and return the context string.


    param input_query (str): The input query string to search for.
    return: The context string retrieved from the vector search.
    rtype: str
    """
    input_vector = generate_embeddings(input_query)
    knn = 3
    vector_field = parse_data_schema()
    vector_query = VectorizedQuery(vector=input_vector, k_nearest_neighbors=knn, fields=vector_field)
    search_client = connect_to_search_index()
    results = search_client.search(
        vector_queries=[vector_query]
    )
    context_map = get_context_map(results)
    context_string = context_map["content"]
    return context_string
def test(input:str) -> str:
    """
    This is a test function to check the toolset functionality.
    
    param input (str): The input string to search for.
    return: The context string retrieved from the vector search.
    rtype: str The dummy context string.
    """
    return "This is dummy content for testing purposes."
def parse_data_schema():
    #TODO: implement this function to parse the data schema and return a dictionary of fields
    #Hard coded for now
    return "contentVector"
def get_context_map(context_raw):
    context_map = {}
    context_map["content"] = ""
    context_map["filenames"] = ""
    for item in context_raw:
        context_map["content"] += item["content"]
        context_map["filenames"] += item["filename"] + ", "
    return context_map

# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    test,
}
