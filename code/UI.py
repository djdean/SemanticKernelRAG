import streamlit as st
from App import App
from Utilities import Utils as Utils
from essential_generators import DocumentGenerator
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from AgentSearchPlugin import AgentSearchPlugin
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread

import time
DO_INIT = True
INSERT_DATA = True
MODEL_FAMILY = "gpt-4"
def main():  
    app = init_app()
    if INSERT_DATA:
        load_data(app)
    
    init_streamlit_session_state(app)
    init_sidebar(app)
    
    run_ui(app)
def run_ui(app):
    st.title("PreRC RAG App")
    st.write("Enter a post below to determine if it will be compliant.")
    messages = st.session_state["messages"]
    history = ""
    for message in messages:
        history += message['content']
        with st.chat_message(message['role']):
                st.markdown(message['content'])
    question = st.chat_input("Enter a possible post :")
    if question:
        handle_question(question, app, history)
def init_semantic_kernel(app):
    kernel = Kernel()
    chat_completion = AzureChatCompletion(
        deployment_name=app.llm_config.config_data["aoai_chat_deployment_name"],
        api_key=app.llm_config.config_data["aoai_key"],
        endpoint=app.llm_config.config_data["aoai_endpoint"],
        api_version=app.llm_config.config_data["aoai_api_version"]
    )
    
    kernel.add_service(chat_completion)
    search_plugin = AgentSearchPlugin(app.vector_db_config.config_data, app.llm_config.config_data, streamlit=True)
    kernel.add_plugin(
        search_plugin,
        plugin_name="Search",
    )
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Required()
    st.session_state["kernel"] = kernel
    st.session_state["execution_settings"] = execution_settings
    st.session_state["chat_completion"] = chat_completion
    history = ChatHistory()
    st.session_state["history"] = history
    return True
def handle_question(question, app, history):
    context_map = None
    question_with_prefix = "Determine if the following post is compliant:\n\n" + question
    #Add the message to the session state
    message = {"role": "user", "content": question_with_prefix}
    st.session_state["messages"].append(message)
    with st.chat_message("user"):
        st.markdown(question_with_prefix)
    # AI Search query - RAG lookup
    if not st.session_state["semantic_kernel"]:   
        with st.spinner("Searching for context..."):    
            question_vector = app.llm_handler.generate_embeddings(question_with_prefix)
            context = app.vector_db_handler.do_vector_search(question_vector)
            context_map = Utils.get_context_map(context)
            context_string = context_map["content"]
        with st.spinner("Answering question..."):    
            response = app.llm_handler.get_response_from_model(context_string, question_with_prefix, history)
    else:
        with st.spinner("Answering question..."):
            asyncio.run(handle_question_semantic_kernel(question_with_prefix))
            while not st.session_state["finished"]:
                time.sleep(5)
            response = st.session_state["response"]
    response_message = {"role": "assistant", "content": response}
    st.session_state["messages"].append(response_message)
    with st.chat_message("assistant"):
        st.markdown(response)
        if not context_map == None and "filenames" in context_map:
            if context_map["filenames"] != "":
                st.markdown("RAG Context: " + context_map["filenames"])
async def handle_question_semantic_kernel(question):
    st.session_state["finished"] = False
    st.session_state["history"].add_user_message(question)
    # Get the response from the AI
    result = await st.session_state["chat_completion"].get_chat_message_content(
        chat_history=st.session_state["history"],
        settings=st.session_state["execution_settings"],
        kernel=st.session_state["kernel"]
    )
    st.session_state["history"].add_message(result)
    st.session_state["response"] = str(result)
    st.session_state["finished"] = True
def init_streamlit_session_state(app):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "history" not in st.session_state:
        st.session_state["history"] = ""
    if "app" not in st.session_state:
        st.session_state["app"] = app
def init_sidebar(app):
    use_semantic_kernel = True
    with st.sidebar:
        use_semantic_kernel = st.checkbox("Use Semantic Kernel", key="init_semantic_kernel",value=True)
    search_index = app.vector_db_config.config_data["search_index_name"]
    with st.sidebar:
        st.header(':green[Azure OpenAI Configuration]')
        st.subheader(':orange[Model:] ' + app.llm_config.config_data["aoai_chat_deployment_name"])
        st.subheader(':orange[API Version:] ' + app.llm_config.config_data["aoai_api_version"])
        st.subheader(':orange[Using Search Index:] ' + search_index)
        with st.spinner("Connectin to AI Search Index: "+search_index+"..."):
            if app.vector_db_handler.connect_to_search_index():
                st.success("Connected to AI Search.")
            else:
                st.error("Failed to connect to AI Search Index, RAG will not work.")
    if use_semantic_kernel:
        st.session_state["semantic_kernel"] = True
        with st.spinner("Initializing Semantic Kernel..."):
            semantic_kernal_status = init_semantic_kernel(app)
            if semantic_kernal_status:
                with st.sidebar:
                    st.success("Initialized Semantic Kernel Successfully.")   
    else:
        st.session_state["semantic_kernel"] = False 
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
def init_app():
    app_config_path = r"C:\Users\dade\Desktop\AzureRAG\config\ai_search_app_syneos.json"
    search_app = App(app_config_path)
    if DO_INIT:
        search_app.do_init()
    return search_app

if __name__ == "__main__":
    main()
