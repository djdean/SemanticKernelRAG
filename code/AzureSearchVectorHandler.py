from VectorDBHandler import VectorDBHandler
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from Config import Config
from Utilities import Utils
from azure.search.documents.models import VectorizedQuery 
from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )

class AzureSearchVectorHandler(VectorDBHandler):
    def __init__(self, config_data):
        self.search_service_endpoint = config_data["search_service_endpoint"]
        self.search_service_key = config_data["search_service_key"]
        self.search_index_name = config_data["search_index_name"]
        self.data_schema = Config(config_data["data_schema_path"]).config_data
        self.parse_data_schema()
        self.credentials = AzureKeyCredential(self.search_service_key)
        self.search_client = None
    def parse_data_schema(self):
        #TODO: implement this function to parse the data schema and return a dictionary of fields
        #Hard coded for now
        self.vector_field = "contentVector"
    def connect_to_search_index(self):
        self.search_client = SearchClient(endpoint=self.search_service_endpoint, 
                                        credential=self.credentials, index_name=self.search_index_name)
        return True
    def connect_to_vector_store(self,config_data):
        self.search_index_client = SearchIndexClient(endpoint=self.search_service_endpoint, 
                                        credential=self.credentials)
        return True
    def reset_db(self, config_data):
        self.connect_to_vector_store(config_data)
        self.search_index_client.delete_index(self.search_index_name)
        return True
    def store_vector_data(self, data):
        if self.search_client is None:
            self.connect_to_search_index()
        if isinstance(data, list):
            self.search_client.upload_documents(documents=data)
        else:
            self.search_client.upload_documents(documents=[data])
        return True
    def do_vector_search(self, input_vector, knn=3, vector_field=None):
        if vector_field is None:
            vector_field = self.vector_field
        vector_query = VectorizedQuery(vector=input_vector, k_nearest_neighbors=knn, fields=vector_field)

        results = self.search_client.search(
            vector_queries=[vector_query]
        )
        return results

    def init_vector_storage(self,config_data):
        self.create_index(config_data)
        #TODO: Add error handling
        return True
    def get_fields_for_schema(self, schema, config_data):
        fields = []
        found_key = False
        found_vector = False
        found_content_key = False
        vector_key = None
        vector_content_key = None
        for key in schema.keys():
            data = schema[key]
            if isinstance(data, str):
                field_type = Utils.parse_schema_string_value(data)
                if field_type == "GUID":
                    found_key = True
                    current_field = SimpleField(name=key, type=SearchFieldDataType.String, key=True)   
                    fields.append(current_field)
                elif field_type == "VECTORCONTENT":
                    vector_content_key = key
                    found_content_key = True
                    current_field = SearchableField(name=key, type=SearchFieldDataType.String)
                    fields.append(current_field)
                else:
                    current_field = SearchableField(name=key, type=SearchFieldDataType.String)
                    fields.append(current_field)
            elif isinstance(data,list):
                found_vector = True
                vector_key = key
                current_field = SearchField(name=key, 
                                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                            searchable=True,
                                            vector_search_dimensions = config_data["vector_dimension"],
                                            vector_search_profile_name = config_data["vector_search_profile"])
                fields.append(current_field)
            elif isinstance(data,int):
                current_field = SimpleField(name=key, type=SearchFieldDataType.Int32)
                fields.append(current_field)
        #Check for keys/content found
        if found_vector:
            self.vector_key = vector_key
        else:
            self.vector_key = "vectorContent"
        if found_content_key:
            self.vector_content_key = vector_content_key
        else:
            self.vector_content_key = "content"
        if found_key:
            self.vector_key = vector_key
        else:
            self.vector_key = "id"
            current_field = SimpleField(name=self.vector_key, type=SearchFieldDataType.String, key=True)   
            fields.append(current_field)

        return fields 
        
    def create_index(self, config_data):
        schema = self.data_schema  
        fields  = self.get_fields_for_schema(schema, config_data)
        search_algo_config_name = config_data["vector_search_algorithm_configuration_name"]
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(name=config_data["vector_search_profile"], 
                                      algorithm_configuration_name=search_algo_config_name)],
            algorithms=[HnswAlgorithmConfiguration(name=search_algo_config_name)]
            )
        index = SearchIndex(name=self.search_index_name, fields=fields, vector_search=vector_search)
        self.search_index_client.create_index(index)
        #TODO: Add error handling
        return True