from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import FunctionTool, ToolSet
from user_functions import user_functions
DO_CREATION = True
def main():
    connection_string = "eastus2.api.azureml.ms;30d3124c-3e71-4697-896f-b402a93e24d4;syneosRAG;dade-1475"
    project_client = connect_to_project(connection_string)
    agent_id = "asst_TtmkEsfFVHStckCoEHWH7nNi"
    if DO_CREATION:
        agent_id = create_agent(project_client).id
    agent = project_client.agents.get_agent(agent_id)
    run_agent(project_client,agent)
def connect_to_project(connection_string):
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=connection_string,
    )
    return project_client
def run_agent(project_client,agent):
    # Create thread for communication
    thread = project_client.agents.create_thread()
    print(f"Created thread, ID: {thread.id}")

    # Create message to thread
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content="Is the following post compliant? \n\n Consider talking to your doctor about Advil for rash treatment. ",
    )
    print(f"Created message, ID: {message.id}")
    # Create and process agent run in thread with tools
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Delete the agent when done
    #project_client.agents.delete_agent(agent.id)
    #print("Deleted agent")

    # Fetch and log all messages
    messages = project_client.agents.list_messages(thread_id=thread.id)
    print(f"Messages: {messages}")
def create_agent(project_client):
    
    functions = FunctionTool(user_functions)
    toolset = ToolSet()
    toolset.add(functions)
    agent = project_client.agents.create_agent(
        model="gpt-4o-mini",
        name="TestAgent",
        instructions="You are an AI agent that is responsible for checking if posts are compliant. The tools you are provided "\
            " with allow you to search for specific information related wether the post is compliant or not.",
        toolset=toolset
    )
       
    print("Agent created successfully")
    print("Agent ID:", agent.id)
    return agent

if __name__ == "__main__":
    main()
