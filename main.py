from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import (
    search_tool, 
    wiki_tool, 
    save_tool,
    pdf_report_tool,
    citation_tool,
    data_extraction_tool,
    arxiv_tool,
    pubmed_tool,
    data_analysis_tool,
    rag_query_tool
)
import os
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str = Field(..., description="Research topic")
    summary: str = Field(..., description="Detailed summary")
    sources: list[str] = Field(..., description="List of sources")
    tools_used: list[str] = Field(..., description="Tools used in research")
    citations: list[str] = Field(default=[], description="Formatted citations")
    charts: list[str] = Field(default=[], description="Generated chart filenames")
    analysis: str = Field(default="", description="Statistical analysis")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro",
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced AI research assistant with these capabilities:
1. Conduct comprehensive literature reviews using academic databases
2. Perform web scraping and data extraction
3. Generate visualizations and statistical analysis
4. Manage citations and references
5. Create professional reports in multiple formats
6. Maintain a knowledge base of previous research

Use available tools and structure output as:\n{format_instructions}"""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [
    search_tool, 
    wiki_tool, 
    save_tool,
    pdf_report_tool,
    citation_tool,
    data_extraction_tool,
    arxiv_tool,
    pubmed_tool,
    data_analysis_tool,
    rag_query_tool
]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    query = input("What can I help you research? ")
    raw_response = agent_executor.invoke({"query": query})
    
    try:
        structured_response = parser.parse(raw_response.get("output")[0]["text"])
        print("\nResearch Results:")
        print(f"Topic: {structured_response.topic}")
        print(f"Summary: {structured_response.summary}")
        print(f"Sources: {structured_response.sources}")
        print(f"Citations: {structured_response.citations}")
        print(f"Charts Generated: {structured_response.charts}")
        print(f"Analysis: {structured_response.analysis}")
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw Response:", raw_response)
