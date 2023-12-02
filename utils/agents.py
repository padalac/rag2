from langchain.chains import RetrievalQA
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import ZeroShotAgent, initialize_agent, Tool, AgentType, AgentExecutor
from langchain.tools import BaseTool
from langchain.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
#from langchain.memory.readonly import ReadOnlySharedMemory

# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
# from langchain.schema import HumanMessage, SystemMessage


def get_tools(rag_qa):
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name = "Enterprise QnA System",
            func=rag_qa,
            description="useful for when you need to answer questions about the company products. Input should be a fully formed question."
        ),
        Tool(
            name = "Backup Google Search",
            func=search.run,
            description="useful for when you need to answer questions but only when the Enterprise QA System couldn't answer the query. Input should be a fully formed question."
        ),
    ]

    return tools

def get_prompt_template(tools):
    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt_template = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    return prompt_template

def get_agent_chain_with_memory(chat_model, prompt_template, tools):
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)

    llm_chain = LLMChain(
        llm=chat_model,
        prompt=prompt_template,
        verbose=True,
        memory=readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
        )

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    return agent_chain