from LLMHandler import LLMHandler
from openai import AzureOpenAI
class AOAIHandler(LLMHandler):
    def __init__(self, config ):
        aoai_endpoint = config["aoai_endpoint"]
        aoai_key = config["aoai_key"]
        aoai_api_version = config["aoai_api_version"]
        self.client = AzureOpenAI(
            azure_endpoint = aoai_endpoint,
            api_key=aoai_key,
            api_version=aoai_api_version
        )
        self.config = config
    def generate_embeddings(self, data, embedding_model=None):
        if embedding_model is None:
            embedding_model = self.config["aoai_embedding_deployment_name"]
        return self.client.embeddings.create(input = [data], model=embedding_model).data[0].embedding
    #TODO: Implement this method
    def get_response_from_model(self, context, question, history,chat_model=None):
        if chat_model is None:
            chat_model = self.config["aoai_chat_deployment_name"]
        response = self.client.chat.completions.create(
            model=chat_model, # model = "deployment_name".
            messages=[
                {"role": "system", "content": "You are an AI assistant extremely proficient in answering questions coming from different users based on input context. Keep your answers short and concise, use bulleted lists when possible. "},
                {"role": "user", "content": "Based on the following context:\n\n"+context+"\n\n Along with the user's chat history:\n\n"+history+"\n\nAnswer the following question:\n\n"+question+" Try to "\
                 "answer the question without the chat history first, then incorporate the chat history into your response if necessary."},
            ]
        )
        return response.choices[0].message.content
