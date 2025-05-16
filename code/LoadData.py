from App import App
from essential_generators import DocumentGenerator
from Utilities import Utils as Utils
DO_INIT = True
INSERT_DATA = True
MODEL_FAMILY = "gpt-4"

def load_data(console_app):
    input_directory = "C:\\Users\\dade\\Desktop\\Syneos\\PreRC\\PreRC regulation documents\\raginputdata"
    file_list = Utils.list_files_in_dir(input_directory)
    data_list = []
    generator = DocumentGenerator()
    for file in file_list:
        filename_only = Utils.get_filename_only(file)
        try:
            with open(input_directory+"\\"+filename_only, mode="r", encoding='utf-8') as f:
                doc = str(f.read())
                chunks = Utils.get_semantic_chunks(doc, MODEL_FAMILY, chunk_size=2000)
                for chunk in chunks:
                    embedding = console_app.llm_handler.generate_embeddings(chunk)
                    id = generator.guid()
                    data = {
                        "id": id,
                        "filename": filename_only,
                        "content": chunk,
                        "contentVector": embedding
                    }
                    data_list.append(data)
        except Exception as e:
            print(f"Error processing file {filename_only}: {e}")
    console_app.insert_data(data_list)

def main():
    app_config_path = r"C:\Users\dade\Desktop\AzureRAG\config\ai_search_app_syneos.json"
    search_app = App(app_config_path)
    if DO_INIT:
        search_app.do_init()
    if INSERT_DATA:
        load_data(search_app)

if __name__ == "__main__":
    main()