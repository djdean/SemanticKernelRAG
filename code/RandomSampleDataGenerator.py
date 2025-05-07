import random, string
from essential_generators import DocumentGenerator
from Utilities import Utils
class RandomSampleDataGenerator():
    def __init__(self, num_samples, schema, llm_handler, random_word_length, random_int_range):
        self.num_samples = num_samples
        self.schema = schema
        self.llm_handler = llm_handler
        self.random_word_length = random_word_length
        self.random_int_range = random_int_range
    def generate_single_sample(self, schema, generator):
        sample = {}
        data = generator.sentence()
        data_vector = self.llm_handler.generate_embeddings(data)  
        letters = string.ascii_lowercase
        set_vector_content = True
        set_content = True
        for key in schema:
            element_data = schema[key]
            if isinstance(element_data, str):
                gentype = Utils.parse_schema_string_value(element_data)
                if gentype == "NAME":
                    sample[key] = generator.name()
                elif gentype == "GUID":
                    sample[key] = generator.guid()
                elif gentype == "EMAIL":
                    sample[key] = generator.email()
                elif gentype == "PHONE":
                    sample[key] = generator.phone()
                elif gentype == "WORD":
                    sample[key] = generator.word()
                elif gentype == "VECTORCONTENT":
                    sample[key] = data
                    set_content = False
                elif gentype == "RANDOM":
                    base_content = ""
                    random_content = ''.join(random.choice(letters) for i in range(self.random_word_length))
                    sample[key] = base_content + random_content
                else:
                    sample[key] = element_data
            elif isinstance(element_data, list):
                sample[key] = data_vector
                set_vector_content = False
            elif isinstance(element_data,int):
                sample[key] = random.randint(0,self.random_int_range)
        if set_content:
            sample["content"] = data
        if set_vector_content:
            sample["contentVector"] = data_vector
        return sample
    def generate_samples(self):
        samples = []
        generator = DocumentGenerator()
        for i in range(self.num_samples):
            sample = self.generate_single_sample(self.schema, generator)
            samples.append(sample)
        return samples