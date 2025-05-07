import json
class Config():
    def __init__(self,config_file_path) -> None:
        self.config_data = self.load_config_data(config_file_path)
    def load_config_data(self,config_file_path):
        with open(config_file_path) as json_file:
            data = json.load(json_file)
        return data