from azure.search.documents.models import VectorizedQuery 
from Config import Config
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI 
from azure.search.documents import SearchClient
from semantic_kernel.functions import kernel_function
from Utilities import Utils as Utils

class AgentSearchPlugin:
    def __init__(self, search_config_data, aoai_config_data, streamlit=False):
        self.search_service_endpoint = search_config_data["search_service_endpoint"]
        self.search_service_key = search_config_data["search_service_key"]
        self.search_index_name = search_config_data["search_index_name"]
        self.data_schema = Config(search_config_data["data_schema_path"]).config_data
        self.parse_data_schema()
        self.credentials = AzureKeyCredential(self.search_service_key)
        self.search_client = None
        connection_result = self.connect_to_search_index()
        if connection_result and not streamlit:
            print("Connected to Azure Search Service successfully")
        elif not connection_result and not streamlit:
            exception_message = "Failed to connect Azure Search Service"
            raise Exception(exception_message)       
        aoai_endpoint = aoai_config_data["aoai_endpoint"]
        aoai_key = aoai_config_data["aoai_key"]
        aoai_api_version = aoai_config_data["aoai_api_version"]
        self.aoai_client = AzureOpenAI(
            azure_endpoint = aoai_endpoint,
            api_key=aoai_key,
            api_version=aoai_api_version
        )
        self.embedding_model = aoai_config_data["aoai_embedding_deployment_name"]
    def connect_to_search_index(self):
        self.search_client = SearchClient(endpoint=self.search_service_endpoint, 
                                        credential=self.credentials, index_name=self.search_index_name)
        return True
    def generate_embeddings(self, data):
        return self.aoai_client.embeddings.create(input = [data], model=self.embedding_model).data[0].embedding
    @kernel_function(name="Search", description="Searches for context using the provided input query.")
    def do_vector_search(self, input_query:str ) -> str:
        """
        Perform a vector search using the input query and return the context string.

        param input_query (str): The input query string to search for.
        return: The context string retrieved from the vector search.
        rtype: str
        """
        input_vector = self.generate_embeddings(input_query)
        knn = 3
        vector_field = self.vector_field
        vector_query = VectorizedQuery(vector=input_vector, k_nearest_neighbors=knn, fields=vector_field)

        results = self.search_client.search(
            vector_queries=[vector_query]
        )
        context_map = Utils.get_context_map(results)
        context_string = context_map["content"]
        return context_string
    def parse_data_schema(self):
        #TODO: implement this function to parse the data schema and return a dictionary of fields
        #Hard coded for now
        self.vector_field = "contentVector"
    
    
