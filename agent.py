# /agent.py

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
import tools as tool_functions_module # Importa o módulo de ferramentas

def initialize_agent(api_key):
    """Configura e inicializa o agente de EDA."""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.0)
    memory = ConversationBufferWindowMemory(k=8, memory_key="chat_history", return_messages=True)
    
    system_prompt = (
        "You are an expert AI assistant specializing in Exploratory Data Analysis (EDA).\n"
        "Your sole task is to analyze the pandas DataFrame named `df`, which is already loaded in the environment.\n\n"
        "**YOUR WORKFLOW:**\n"
        "1.  **Understand First:** When the user asks a broad question, ALWAYS start by using the `get_data_summary` tool.\n"
        "2.  **Choose the Right Tool and Synthesize:** Based on your understanding and the user's query, select the most appropriate tool. After the tool runs, you will receive its output.\n\n"
        "**ESSENTIAL RULES:**\n"
        "1.  **The `df` DataFrame always exists.** Never ask the user to load or provide data.\n"
        "2.  **CRITICAL RULE FOR OUTPUT:** Your final answer to the user MUST BE a direct and complete synthesis of the `message` field returned by the tool. **NEVER just say 'it was successful'. ALWAYS present the numbers, tables, and findings from the tool's message in your final response.** If the tool returns a table in its message, you must show that information.\n"
        "3.  **LANGUAGE:** You **MUST** provide your final response to the user in Brazilian Portuguese (Português do Brasil)."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Pega as funções do módulo de ferramentas
    tool_functions = [
    tool_functions_module.get_data_summary,
    tool_functions_module.handle_missing_values,
    tool_functions_module.handle_duplicates,
    tool_functions_module.change_column_type,
    tool_functions_module.create_feature_from_math,
    tool_functions_module.bin_numerical_column,
    tool_functions_module.plot_histogram,
    tool_functions_module.plot_bar_chart,
    tool_functions_module.plot_scatter,
    tool_functions_module.plot_correlation_heatmap,
    tool_functions_module.plot_word_cloud,
    tool_functions_module.plot_pair_plot,
    tool_functions_module.get_aggregated_data,
    tool_functions_module.create_pivot_table,
    tool_functions_module.perform_t_test,
    tool_functions_module.detect_outliers,
    tool_functions_module.export_dataframe
    ]

    tools_as_langchain = [Tool(name=fn.__name__, description=fn.__doc__, func=fn) for fn in tool_functions]
    agent_logic = create_tool_calling_agent(llm, tools_as_langchain, prompt)

    return AgentExecutor(
        agent=agent_logic, tools=tools_as_langchain, verbose=True, memory=memory,
        max_iterations=15, handle_parsing_errors=True
    )