import os
import tiktoken
import json
import semchunk
class Utils:
    def __init__(self):
        pass
    def get_file_name_only(self, file_name):
        file_name_split = file_name.split("/")
        file_name_with_extension = file_name_split[len(file_name_split)-1]
        file_name_with_extension_split = file_name_with_extension.split(".")
        file_name_only = file_name_with_extension_split[0]
        return file_name_only
    @staticmethod
    def get_semantic_chunks(text, model, chunk_size=8000):
        encoder = tiktoken.encoding_for_model(model)
        token_counter = lambda text: len(encoder.encode(text))
        chunks = semchunk.chunk(text, chunk_size=chunk_size, token_counter=token_counter)
        return chunks
    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    @staticmethod
    def parse_schema_string_value(element_data):
        gentype = None
        str_split = element_data.split("|")
        if len(str_split) > 1:
            gentype = str_split[1]
        return gentype
    @staticmethod
    def list_files_in_dir(path):
        return os.listdir(path)
    @staticmethod
    def read_json_data(file_name):
        with open(file_name) as json_file:
            data = json.load(json_file)
        return data
    @staticmethod
    def get_filename_only(filename_with_path):
        return filename_with_path.split("/")[-1]
    @staticmethod
    def get_filename_windows_only(filename_with_path):
        return filename_with_path.split("\\")[-1]
    @staticmethod
    def get_file_without_extension(filename):
        return filename.split(".")[0]
    @staticmethod
    def get_context_map(context_raw):
        context_map = {}
        context_map["content"] = ""
        context_map["filenames"] = ""
        for item in context_raw:
            context_map["content"] += item["content"]
            context_map["filenames"] += item["filename"] + ", "
        return context_map
