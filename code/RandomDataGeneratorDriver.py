from RandomSampleDataGenerator import RandomSampleDataGenerator
from Config import Config
from AOAIHandler import AOAIHandler
def main():
    AOAI_config_path = r"C:\Users\dade\Desktop\AzureRAG\config\aoai_config.json"
    config = Config(AOAI_config_path)
    aoai_endpoint = config.config_data["aoai_endpoint"]
    aoai_key = config.config_data["aoai_key"]
    aoai_api_version = config.config_data["aoai_api_version"]
    aoai_temperature = config.config_data["aoai_temperature"]
    aoai_model = config.config_data["aoai_deployment_name"]
    aoai_handler = AOAIHandler(aoai_endpoint, aoai_key, aoai_api_version, aoai_temperature, aoai_model)
    num_samples = 10
    schema = {
        "name": "Jon|NAME",
        "age": 25,
        "company": "Microsoft|WORD",
        "phone": "|PHONE",
        "email": "|EMAIL",
        "Other": "Random|RANDOM"
    }
    random_word_length = 10
    random_int_range = 100
    random_data_generator = RandomSampleDataGenerator(num_samples, schema, aoai_handler, random_word_length, random_int_range)
    samples = random_data_generator.generate_samples()
    for sample in samples:
        print(sample)
if __name__ == "__main__":
    main()