from CosmosDBPyMongoVectorHandler import CosmosDBPyMongoVectorHandler
from LLMHandler import LLMHandler
from AOAIHandler import AOAIHandler
from VectorDBHandler import VectorDBHandler
from AzureSearchVectorHandler import AzureSearchVectorHandler
from Config import Config
from typing import Tuple
class App:
    def __init__(self,app_config_path):
        self.app_config = Config(app_config_path)
        (self.vector_db_handler,self.vector_db_config) = self.init_vector_db(self.app_config)
        (self.llm_handler,self.llm_config) = self.init_LLM(self.app_config)   
    def insert_data(self, data):
        self.vector_db_handler.store_vector_data(data)
    def do_init(self):
        self.vector_db_handler.reset_db(self.vector_db_config.config_data)
        self.vector_db_handler.init_vector_storage(self.vector_db_config.config_data)
        return True
    def run_test(self):
        test_query = "What is a test?"
        test_vector = self.llm_handler.generate_embeddings(test_query)
        results = self.vector_db_handler.do_vector_search(test_vector,self.app_config.config_data)
        for result in results:
            for key in result.keys():
                if key != self.vector_db_handler.vector_key:
                    print(key +": "+ str(result[key]))

    def init_vector_db(self,app_config) -> Tuple[VectorDBHandler,Config]: 
        vector_db_handler = None
        vector_db_config = None
        vector_db_config_path = app_config.config_data["vector_db_config_path"]
        if (app_config.config_data["vector_storage_mode"] == "COSMOS"):
            vector_db_config = Config(vector_db_config_path)
            vector_db_handler =  self.init_cosmos_db(vector_db_config)
        elif (app_config.config_data["vector_storage_mode"] == "COGSEARCH"):
            vector_db_config = Config(vector_db_config_path)
            vector_db_handler =  self.init_cogsearch(vector_db_config)
        return (vector_db_handler, vector_db_config)

    def init_LLM(self,app_config)->Tuple[LLMHandler,Config]:
        llm_handler = None
        llm_config = None
        llm_config_path = app_config.config_data["llm_config_path"]
        if (app_config.config_data["LLM"] == "AOAI"):
            llm_config = Config(llm_config_path)
            llm_handler = self.init_AOAI(llm_config)
        return (llm_handler, llm_config)

    def init_AOAI(self,config):
        aoai_handler = AOAIHandler(config.config_data)
        return aoai_handler
    def init_cosmos_db(self,config):
        cosmos_vector_handler = CosmosDBPyMongoVectorHandler(config.config_data)
        return cosmos_vector_handler
    def init_cogsearch(self,config):
        ai_search_handler = AzureSearchVectorHandler(config.config_data)
        return ai_search_handler
    