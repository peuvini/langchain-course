import os
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()
@tool
def search(query: str) -> str:
    """Search for information on the internet."""
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5-nano")
tools = [search]
agent = create_agent(model=llm,tools=tools)

def main():
    result = agent.invoke({
        "messages":[
            HumanMessage(content="search for 3 job postings for an ai engineer using langchain in france on linkedin and list their details")
            ]
    })
    print(result)

if __name__ == "__main__":
    main()