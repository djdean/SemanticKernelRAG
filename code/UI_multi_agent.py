import streamlit as st
from App import App
from Utilities import Utils as Utils

import asyncio

from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from AgentSearchPlugin import AgentSearchPlugin
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent



def main():  
    app = init_app()
    
    
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
def init_agents(kernel,plugin):
    more_info_keyword = "done"
    ready_keyword = "ready"
    finished_keyword = "finished"
    st.session_state["more_info_keyword"] = more_info_keyword
    st.session_state["ready_keyword"] = ready_keyword
    st.session_state["finished_keyword"] = finished_keyword
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    # Create the agent
    reviewer_agent = ChatCompletionAgent(
        kernel=kernel,
        name="reviewer",
        instructions=f"""
            You are responsible for checking if all the information necessary to determine if the post is compliant is present in the post.
            If all the information is not present, provide a list of the missing information and reply '{more_info_keyword}'. If all the information is present,
            mention all content is available, provide a list of the information and reply '{ready_keyword}'. Be polite in your response. Always reply with a keyword
            either '{more_info_keyword}' or '{ready_keyword}' at the end of your response.
            
            The information you need to check is:
                1.	The pharma or biotech company name.
                2.	The drug brand name. If there is no brand name, say there is no brand name. If no drug name is provided, check if the content is meant to support an investigational drug.
                3.  The treatment indication of the drug.
                4.  If the drug is the only drug available for the treatment indication. (yes or no)
                5.  The social media platform where the post will be published.
                6.  The handle the post will be published under. Examples are: corporte handle, employee handle, brand handle, influencer handle, etc.
                7.  The date the post will be published.
                8.  The post content.
            Reply using the following format:
                1. Company Name: <company_name>
                2. Drug Brand Name: <drug_brand_name>
                3. Treatment Indication: <treatment_indication>
                4. Is the drug the only drug available for the treatment indication? <yes or no>
                5. Social Media Platform: <social_media_platform>
                6. Post Handle: <handle>
                7. Post Date: <post_date>
                8. Post Content: <post_content>
            """,
        arguments=KernelArguments(settings=execution_settings),
    )
    compliance_cheking_agent = ChatCompletionAgent(
        kernel=kernel,
        name="checker",
        instructions=f"""
            Your responsibility is to check if a provided post is compliant or not. Use the provided 'ComplianceSearch' plugin to 
            search for relevant context with the provided list of information and use that context to determine if the post is compliant or not. Reply with 
            your answer along with a rationalle for your answer. Always reply '{finished_keyword}' when finished.
            """,
        arguments=KernelArguments(settings=execution_settings),
        plugins=[plugin]
    )
    editing_agent = ChatCompletionAgent(
        kernel=kernel,
        name="editor",
        instructions=f"""
            Your responsibility is to use the chat history to apply the necessary edits to the post. Once the edits are done, reply 
            with '{ready_keyword}'. Always be sure to reply '{ready_keyword}' when finished.
            """,
        arguments=KernelArguments(settings=execution_settings),
    )
    selection_function = KernelFunctionFromPrompt(
        function_name="selection", 
        prompt=f"""
            Examine the provided RESPONSE and choose the next participant.
            State only the name of the chosen participant without explanation.
            Never choose the participant named in the RESPONSE.

            Choose only from these participants:
            - reviewer
            - checker
            - editor

            Rules:
            - If RESPONSE is user input, it is reviewer's turn.
            - If RESPONSE contains the keyword '{ready_keyword}', it is checker's turn.
            - If RESPONSE asks about editing, correcting, or changing something, it's editor's turn.
            - If it was just editor's turn, it is reviewer's turn.
            RESPONSE:
            {{{{$lastmessage}}}}
            """
        )
    
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
            Examine the RESPONSE and determine whether we need more information from the user or not.
            When the RESPONSE contains the keyword '{more_info_keyword}', reply with a single word: {more_info_keyword}
            When the RESPONSE contains the keyword '{finished_keyword}', reply with a single word: {finished_keyword}
            When the RESPONSE contains the keyword '{ready_keyword}', reply with a single word: {ready_keyword}
            RESPONSE:
            {{{{$lastmessage}}}}
            """,
    )
    if not "agent_chat" in st.session_state or st.session_state["agent_chat"] == None: 
        history_reducer = ChatHistoryTruncationReducer(target_count=5)
        # Create the AgentGroupChat with selection and termination strategies.
        chat = AgentGroupChat(
            agents=[reviewer_agent, compliance_cheking_agent, editing_agent],
            selection_strategy=KernelFunctionSelectionStrategy(
                initial_agent=reviewer_agent,
                function=selection_function,
                kernel=kernel,
                result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else "reviewer",
                history_variable_name="lastmessage",
                history_reducer=history_reducer,
            ),
            termination_strategy=KernelFunctionTerminationStrategy(
                agents=[reviewer_agent,compliance_cheking_agent],
                function=termination_function,
                kernel=kernel,
                result_parser=lambda result: (finished_keyword in str(result.value[0]).lower()) or (more_info_keyword in str(result.value[0]).lower()),
                history_variable_name="lastmessage",
                maximum_iterations=10,
                history_reducer=history_reducer,
            ),
        )
        st.session_state["agent_chat"] = chat
def init_semantic_kernel(app):
    kernel = Kernel()
    chat_completion = AzureChatCompletion(
        deployment_name=app.llm_config.config_data["aoai_chat_deployment_name"],
        api_key=app.llm_config.config_data["aoai_key"],
        endpoint=app.llm_config.config_data["aoai_endpoint"],
        api_version=app.llm_config.config_data["aoai_api_version"]
    )
    st.session_state["chat_completion"] = chat_completion
    kernel.add_service(chat_completion)
    search_plugin = AgentSearchPlugin(app.vector_db_config.config_data, app.llm_config.config_data, streamlit=True)
    init_agents(kernel,search_plugin)
    st.session_state["kernel"] = kernel
    return True
def handle_question(user_input):
    #Add the message to the session state
    message = {"role": "user", "content": user_input}
    st.session_state["messages"].append(message)
    with st.chat_message("user"):
        st.markdown(user_input)
    handle_question_semantic_kernel(user_input)
def handle_question_semantic_kernel(user_input):
    with st.spinner("Answering question..."):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(get_answer_from_agents(user_input))
        loop.close()

async def get_answer_from_agents(question):
    chat = st.session_state["agent_chat"]
    # Add the current user_input to the chat
    await chat.add_chat_message(message=question)
    try:
        async for response in chat.invoke():
            if response is None or not response.name:
                continue
            response_message = {"role": response.name, "content": response.content}
            st.session_state["messages"].append(response_message)
            with st.chat_message(response_message['role']):
                st.markdown(response_message['content'])
            if st.session_state["more_info_keyword"] in response.content.lower():
                continue
            elif st.session_state["finished_keyword"] in response.content.lower():
                continue
    except Exception as e:
        print(f"Error during chat invocation: {e}")
    chat.is_complete = False
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
def reset_chat():
    if st.session_state["messages"] != None:
        st.session_state["messages"] = []
        st.session_state["agent_chat"] = None

def init_app():
    app_config_path = r"C:\Users\dade\Desktop\AzureRAG\config\ai_search_app_syneos.json"
    search_app = App(app_config_path)
    return search_app

if __name__ == "__main__":
    main()
