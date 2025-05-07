from App import App
from RandomSampleDataGenerator import RandomSampleDataGenerator
from Config import Config
def main():
    app_config_path = r"C:\Users\dade\Desktop\AzureRAG\config\ai_search_app_config.json"
    random_data_config_path = r"C:\Users\dade\Desktop\AzureRAG\config\random_data_config.json"
    random_data_config = Config(random_data_config_path)
    DO_INIT = True
    INSERT_DATA = True
    RANDOMIZE_DATA = True
    console_app = App(app_config_path)
    if DO_INIT:
        console_app.do_init()
    else:
        console_app.vector_db_handler.connect_to_vector_store(console_app.vector_db_config.config_data)
    if INSERT_DATA:
        if RANDOMIZE_DATA:
            schema = Config(random_data_config.config_data["schema_path"]).config_data
            num_samples = random_data_config.config_data["num_samples"]
            random_word_length = random_data_config.config_data["random_word_length"]
            random_int_range = random_data_config.config_data["random_int_range"]
            random_data_generator = RandomSampleDataGenerator(num_samples, schema, console_app.llm_handler, random_word_length, random_int_range)
            samples = random_data_generator.generate_samples()
            console_app.insert_data(samples)
        else:
            data = "This is a test"
            vector = console_app.llm_handler.generate_embeddings(data)
            content_to_store = {
                "content": data,
                "contentVector": vector,
                "model": console_app.llm_config.config_data["aoai_deployment_name"],
                "company": "Microsoft"
            }     
            console_app.insert_data(content_to_store)
    console_app.run_test()
if __name__ == "__main__":
    main()